import * as vscode from 'vscode';
import { endpoint } from './models';
import { out } from './extension';

type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

let _provider: LlamaCppProvider | undefined;

export function notifyProviderConfigChanged(): void {
    _provider?.notifyConfigChanged();
}

export function registerLmProvider(context: vscode.ExtensionContext): void {
    // @llama chat participant — works in all VS Code versions
    const participant = vscode.chat.createChatParticipant('llamacpp.assistant', handleChat);
    participant.iconPath = new vscode.ThemeIcon('sparkle');
    context.subscriptions.push(participant);

    // Language model provider — registers models in Copilot Chat's model picker
    const hasRegister = typeof (vscode.lm as any).registerLanguageModelChatProvider === 'function';
    const hasTextPart = typeof (vscode as any).LanguageModelTextPart === 'function';
    out.appendLine(`[setup] registerLanguageModelChatProvider: ${hasRegister}  LanguageModelTextPart: ${hasTextPart}`);

    _provider = new LlamaCppProvider();
    if (hasRegister) {
        const reg = (vscode.lm as any).registerLanguageModelChatProvider('llamacpp', _provider);
        context.subscriptions.push(reg);
        out.appendLine('[setup] Language model provider registered — models will appear in Copilot Chat picker');
    } else {
        out.appendLine('[setup] registerLanguageModelChatProvider not available — use @llama participant instead');
    }
}

// ---------------------------------------------------------------------------
// Language model provider (Copilot Chat model picker)
// ---------------------------------------------------------------------------

class LlamaCppProvider {
    private _emitter = new vscode.EventEmitter<void>();
    readonly onDidChange = this._emitter.event;

    notifyConfigChanged(): void {
        this._emitter.fire();
    }

    async provideLanguageModelChatInformation(_options: any, _token: vscode.CancellationToken): Promise<any[]> {
        const ep = endpoint();
        try {
            const [modelsRes, propsRes] = await Promise.allSettled([
                fetch(`${ep}/v1/models`, { signal: AbortSignal.timeout(5000) }),
                fetch(`${ep}/props`,     { signal: AbortSignal.timeout(3000) }),
            ]);

            let contextSize = 16384;
            if (propsRes.status === 'fulfilled' && propsRes.value.ok) {
                const props: any = await propsRes.value.json();
                const n_ctx = props.default_generation_settings?.n_ctx;
                if (n_ctx && n_ctx >= 1000) { contextSize = n_ctx; }
            }

            if (modelsRes.status === 'rejected' || !modelsRes.value.ok) { return []; }
            const json: any = await modelsRes.value.json();
            const models: { id: string }[] = json.data ?? [];

            const maxOutput = Math.min(4096, Math.floor(contextSize / 2));
            const maxInput  = Math.max(1, contextSize - maxOutput);

            out.appendLine(`[${ts()}] provider: found ${models.length} model(s) — ${models.map(m => m.id).join(', ')}`);

            return models.map(m => ({
                id:              m.id,
                name:            m.id,
                tooltip:         `Llama.cpp — ${m.id}`,
                family:          'llama-cpp',
                version:         '1.0.0',
                maxInputTokens:  maxInput,
                maxOutputTokens: maxOutput,
                capabilities: {
                    toolCalling: true,
                    imageInput:  false,
                },
                isUserSelectable: true,
                metadata: {},
            }));
        } catch (err) {
            out.appendLine(`[${ts()}] provider: model discovery failed — ${(err as Error).message}`);
            return [];
        }
    }

    async provideTokenCount(
        _model: any,
        text: string | vscode.LanguageModelChatMessage,
        _token: vscode.CancellationToken
    ): Promise<number> {
        const content = typeof text === 'string'
            ? text
            : typeof (text as any).content === 'string'
                ? (text as any).content
                : Array.isArray((text as any).content)
                    ? (text as any).content.map((p: any) => p.value ?? p.text ?? '').join('')
                    : String((text as any).content ?? '');

        try {
            const res = await fetch(`${endpoint()}/tokenize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content }),
                signal: AbortSignal.timeout(2000),
            });
            if (res.ok) {
                const json: any = await res.json();
                return (json.tokens ?? []).length;
            }
        } catch { /* fall through */ }

        return Math.ceil(content.length / 4);
    }

    async provideLanguageModelChatResponse(
        model: any,
        messages: readonly any[],
        options: any,
        progress: any,
        token: vscode.CancellationToken
    ): Promise<void> {
        const mapped = convertToOpenAI(messages);
        const maxTokens: number = options?.modelOptions?.max_tokens ?? 4096;

        out.appendLine(`[${ts()}] lm-provider  model=${model.id}  messages=${mapped.length}  maxTokens=${maxTokens}`);

        const controller = new AbortController();
        token.onCancellationRequested(() => controller.abort());

        const ep = endpoint();
        out.appendLine(`[${ts()}] → POST ${ep}/v1/chat/completions  model=${model.id}`);
        const t0 = Date.now();

        const res = await fetch(`${ep}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model.id, messages: mapped, max_tokens: maxTokens, stream: true }),
            signal: controller.signal,
        });

        if (!res.ok || !res.body) {
            throw new Error(`llama.cpp server returned ${res.status}: ${await res.text()}`);
        }

        out.appendLine(`[${ts()}] ← ${res.status} streaming…`);

        const TextPart = (vscode as any).LanguageModelTextPart;
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let chars = 0;
        let firstChunk = true;

        try {
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
                        return;
                    }
                    let text = '';
                    try {
                        const chunk = JSON.parse(data);
                        text = chunk.choices?.[0]?.delta?.content ?? '';
                    } catch { continue; }

                    if (!text) { continue; }

                    if (firstChunk) {
                        firstChunk = false;
                        out.appendLine(`[${ts()}] ← first token  TextPart=${!!TextPart}`);
                    }

                    try {
                        progress.report(TextPart ? new TextPart(text) : { value: text });
                        chars += text.length;
                    } catch (reportErr) {
                        out.appendLine(`[${ts()}] progress.report error: ${(reportErr as Error).message}`);
                        throw reportErr;
                    }
                }
            }
        } catch (err) {
            if ((err as Error).name === 'AbortError' || token.isCancellationRequested) {
                out.appendLine(`[${ts()}] ← stream cancelled  ${chars} chars  ${Date.now() - t0}ms`);
                return;
            }
            throw err;
        }
    }
}

// ---------------------------------------------------------------------------
// @llama chat participant
// ---------------------------------------------------------------------------

async function handleChat(
    request: vscode.ChatRequest,
    chatContext: vscode.ChatContext,
    response: vscode.ChatResponseStream,
    token: vscode.CancellationToken
): Promise<vscode.ChatResult> {
    const cfg = vscode.workspace.getConfiguration('llamacpp');
    const maxTokens = cfg.get<number>('chatMaxTokens', 2048);
    const preview = request.prompt.slice(0, 80).replace(/\n/g, ' ');

    out.appendLine(`[${ts()}] @llama  maxTokens=${maxTokens}  history=${chatContext.history.length}  prompt="${preview}${request.prompt.length > 80 ? '…' : ''}"`);

    const history = chatContext.history
        .filter((_h): _h is vscode.ChatRequestTurn | vscode.ChatResponseTurn => true)
        .flatMap<ChatMessage>(h => {
            if (h instanceof vscode.ChatRequestTurn) {
                return [{ role: 'user', content: h.prompt }];
            }
            const text = h.response
                .flatMap(r => r instanceof vscode.ChatResponseMarkdownPart ? [r.value.value] : [])
                .join('');
            return text ? [{ role: 'assistant', content: text }] : [];
        });

    const messages: ChatMessage[] = [
        { role: 'system', content: 'You are an expert coding assistant. Be concise and accurate.' },
        ...history,
        { role: 'user', content: request.prompt },
    ];

    try {
        const chars = await streamToParticipant(messages, maxTokens, response, token);
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

async function streamToParticipant(
    messages: ChatMessage[],
    maxTokens: number,
    response: vscode.ChatResponseStream,
    token: vscode.CancellationToken
): Promise<number> {
    const controller = new AbortController();
    token.onCancellationRequested(() => controller.abort());

    const ep = endpoint();
    out.appendLine(`[${ts()}] → POST ${ep}/v1/chat/completions`);

    const res = await fetch(`${ep}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, max_tokens: maxTokens, stream: true }),
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
                out.appendLine(`[${ts()}] ← [DONE]  ${chars} chars`);
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function convertToOpenAI(messages: readonly any[]): ChatMessage[] {
    // VS Code LM roles are either strings ('user','assistant','system')
    // or numeric enum values (User=1, Assistant=2, System=3)
    const roleMap: Record<string, 'user' | 'assistant' | 'system'> = {
        user: 'user', '1': 'user',
        assistant: 'assistant', '2': 'assistant',
        system: 'system', '3': 'system',
    };
    return messages.map(m => ({
        role: roleMap[String(m.role)] ?? 'user',
        content: typeof m.content === 'string'
            ? m.content
            : Array.isArray(m.content)
                ? m.content.map((p: any) => p.value ?? p.text ?? '').join('')
                : String(m.content ?? ''),
    }));
}

function ts(): string {
    return new Date().toLocaleTimeString();
}
