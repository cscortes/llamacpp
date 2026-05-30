import * as vscode from 'vscode';
import { endpoint, activeModel } from './models';
import { out } from './extension';

type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

export function registerLmProvider(context: vscode.ExtensionContext): void {
    const participant = vscode.chat.createChatParticipant('llamacpp.assistant', handleChat);
    participant.iconPath = new vscode.ThemeIcon('sparkle');
    context.subscriptions.push(participant);

    // Proposed API: register as a selectable model in the Copilot model picker.
    // Requires "enabledApiProposals": ["languageModels"] in package.json
    // and VS Code >= 1.90 with the proposal active.
    if ('registerChatModelProvider' in vscode.lm) {
        for (const model of ['phi', 'qwen2', 'deep']) {
            const disposable = (vscode.lm as any).registerChatModelProvider(
                `llamacpp/${model}`,
                makeLmHandler(model),
                {
                    vendor: 'llamacpp',
                    family: model,
                    version: '1.0',
                    maxInputTokens: 16384,
                    maxOutputTokens: 4096,
                }
            );
            context.subscriptions.push(disposable);
        }
    }
}

async function handleChat(
    request: vscode.ChatRequest,
    chatContext: vscode.ChatContext,
    response: vscode.ChatResponseStream,
    token: vscode.CancellationToken
): Promise<vscode.ChatResult> {
    const cfg = vscode.workspace.getConfiguration('llamacpp');
    const maxTokens = cfg.get<number>('chatMaxTokens', 2048);
    const model = activeModel();
    const preview = request.prompt.slice(0, 80).replace(/\n/g, ' ');

    out.appendLine(`[${ts()}] @llama  model=${model.id}  maxTokens=${maxTokens}  history=${chatContext.history.length}  prompt="${preview}${request.prompt.length > 80 ? '…' : ''}"`);

    const history = chatContext.history
        .filter((_h): _h is vscode.ChatRequestTurn | vscode.ChatResponseTurn => true)
        .flatMap<ChatMessage>(h => {
            if (h instanceof vscode.ChatRequestTurn) {
                return [{ role: 'user', content: h.prompt }];
            }
            const text = h.response.flatMap(r => r instanceof vscode.ChatResponseMarkdownPart ? [r.value.value] : []).join('');
            return text ? [{ role: 'assistant', content: text }] : [];
        });

    const messages: ChatMessage[] = [
        { role: 'system', content: 'You are an expert coding assistant. Be concise and accurate.' },
        ...history,
        { role: 'user', content: request.prompt },
    ];

    try {
        const chars = await streamChat(messages, maxTokens, model.id, response, token);
        out.appendLine(`[${ts()}] @llama  done  returned=${chars} chars`);
    } catch (err) {
        if ((err as Error).name === 'AbortError') {
            out.appendLine(`[${ts()}] @llama  cancelled`);
        } else {
            out.appendLine(`[${ts()}] @llama  error: ${(err as Error).message}`);
            response.markdown(`\n\n**Error:** ${(err as Error).message}`);
        }
    }

    return {};
}

function makeLmHandler(modelId: string) {
    return {
        async provideLanguageModelResponse(
            messages: any[],
            options: any,
            token: vscode.CancellationToken,
            progress: any
        ) {
            const maxTokens = options.maxTokens ?? 2048;
            out.appendLine(`[${ts()}] lm:${modelId}  messages=${messages.length}  maxTokens=${maxTokens}`);

            const mapped: ChatMessage[] = messages.map((m: any) => ({
                role: m.role,
                content: typeof m.content === 'string'
                    ? m.content
                    : m.content.map((p: any) => p.value ?? '').join(''),
            }));
            const sink = { markdown: (t: string) => progress.report({ index: 0, part: { value: t } }) };

            try {
                const chars = await streamChat(mapped, maxTokens, modelId, sink as any, token);
                out.appendLine(`[${ts()}] lm:${modelId}  done  returned=${chars} chars`);
            } catch (err) {
                out.appendLine(`[${ts()}] lm:${modelId}  ${(err as Error).name === 'AbortError' ? 'cancelled' : 'error: ' + (err as Error).message}`);
            }
        }
    };
}

async function streamChat(
    messages: ChatMessage[],
    maxTokens: number,
    modelId: string,
    response: { markdown: (s: string) => void },
    token: vscode.CancellationToken
): Promise<number> {
    const controller = new AbortController();
    token.onCancellationRequested(() => controller.abort());

    const t0 = Date.now();
    out.appendLine(`[${ts()}] → POST ${endpoint()}/v1/chat/completions  model=${modelId}`);

    const res = await fetch(`${endpoint()}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId, messages, max_tokens: maxTokens, stream: true }),
        signal: controller.signal,
    });

    if (!res.ok || !res.body) {
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
    }

    out.appendLine(`[${ts()}] ← ${res.status} streaming…`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let chars = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) { break; }
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
            if (!line.startsWith('data: ')) { continue; }
            const data = line.slice(6).trim();
            if (data === '[DONE]') {
                out.appendLine(`[${ts()}] ← [DONE]  ${chars} chars  ${Date.now() - t0}ms`);
                return chars;
            }
            try {
                const chunk = JSON.parse(data);
                const text: string = chunk.choices?.[0]?.delta?.content ?? '';
                if (text) { response.markdown(text); chars += text.length; }
            } catch { /* skip malformed chunk */ }
        }
    }

    return chars;
}

function ts(): string {
    return new Date().toLocaleTimeString();
}
