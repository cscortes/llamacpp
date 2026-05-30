import * as vscode from 'vscode';
import { endpoint, activeModel } from './models';
import { out } from './extension';

const DEBOUNCE_MS = 400;
const PREFIX_CHARS = 2000;
const SUFFIX_CHARS = 500;

export function registerInlineCompletions(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            new LlamaCppCompletionProvider()
        )
    );
}

class LlamaCppCompletionProvider implements vscode.InlineCompletionItemProvider {
    private lastTimer: ReturnType<typeof setTimeout> | undefined;
    private lastController: AbortController | undefined;

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        _context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionList | undefined> {
        const cfg = vscode.workspace.getConfiguration('llamacpp');
        if (!cfg.get<boolean>('inlineCompletions', true)) { return; }

        await this.debounce();
        if (token.isCancellationRequested) { return; }

        this.lastController?.abort();
        const controller = new AbortController();
        this.lastController = controller;
        token.onCancellationRequested(() => controller.abort());

        const offset = document.offsetAt(position);
        const text = document.getText();
        const prefix = text.slice(Math.max(0, offset - PREFIX_CHARS), offset);
        const suffix = text.slice(offset, offset + SUFFIX_CHARS);
        const maxTokens = cfg.get<number>('inlineMaxTokens', 128);
        const model = activeModel();
        const file = document.fileName.split(/[\\/]/).pop() ?? document.fileName;
        const loc = `${file}:${position.line + 1}:${position.character + 1}`;
        const method = model.supportsFim ? 'infill' : 'chat';

        out.appendLine(`[${ts()}] inline  ${loc}  model=${model.id}  ${method}  prefix=${prefix.length}ch  suffix=${suffix.length}ch`);
        const t0 = Date.now();

        try {
            const completion = model.supportsFim
                ? await fetchInfill(prefix, suffix, maxTokens, model.id, controller.signal)
                : await fetchChatCompletion(prefix, suffix, maxTokens, model.id, controller.signal);

            if (token.isCancellationRequested) {
                out.appendLine(`[${ts()}] inline  ${loc}  cancelled after ${Date.now() - t0}ms`);
                return;
            }

            if (!completion) {
                out.appendLine(`[${ts()}] inline  ${loc}  empty  ${Date.now() - t0}ms`);
                return;
            }

            out.appendLine(`[${ts()}] inline  ${loc}  → ${completion.length} chars  ${Date.now() - t0}ms`);
            return {
                items: [new vscode.InlineCompletionItem(completion, new vscode.Range(position, position))],
            };
        } catch (err) {
            if ((err as Error).name !== 'AbortError') {
                out.appendLine(`[${ts()}] inline  ${loc}  error: ${(err as Error).message}`);
            }
            return;
        }
    }

    private debounce(): Promise<void> {
        return new Promise(resolve => {
            clearTimeout(this.lastTimer);
            this.lastTimer = setTimeout(resolve, DEBOUNCE_MS);
        });
    }
}

function ts(): string {
    return new Date().toLocaleTimeString();
}

async function fetchInfill(
    prefix: string,
    suffix: string,
    maxTokens: number,
    modelId: string,
    signal: AbortSignal
): Promise<string> {
    const res = await fetch(`${endpoint()}/infill`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input_prefix: prefix,
            input_suffix: suffix,
            n_predict: maxTokens,
            temperature: 0.1,
            stop: ['\n\n', '```'],
        }),
        signal,
    });
    if (!res.ok) { return ''; }
    const json: any = await res.json();
    return json.content ?? '';
}

async function fetchChatCompletion(
    prefix: string,
    suffix: string,
    maxTokens: number,
    modelId: string,
    signal: AbortSignal
): Promise<string> {
    const prompt = `Complete the following code. Output only the completion, no explanation.\n\`\`\`\n${prefix}<FILL_HERE>${suffix}\n\`\`\``;
    const res = await fetch(`${endpoint()}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: modelId,
            messages: [
                { role: 'system', content: 'You are a code completion engine. Output only the missing code, nothing else.' },
                { role: 'user', content: prompt },
            ],
            max_tokens: maxTokens,
            temperature: 0.1,
            stop: ['\n\n', '```'],
            stream: false,
        }),
        signal,
    });
    if (!res.ok) { return ''; }
    const json: any = await res.json();
    return json.choices?.[0]?.message?.content ?? '';
}
