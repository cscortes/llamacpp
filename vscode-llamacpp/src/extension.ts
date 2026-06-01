import * as vscode from 'vscode';
import { MODELS } from './models';
import { registerLmProvider } from './lmProvider';
import { registerInlineCompletions } from './completions';
import { registerServerCheck } from './serverCheck';

export let out: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext): void {
    out = vscode.window.createOutputChannel('llama.cpp Local AI');
    context.subscriptions.push(out);

    out.appendLine('llama.cpp Local AI activating…');

    registerServerCheck(context);
    registerLmProvider(context);
    registerInlineCompletions(context);

    context.subscriptions.push(
        vscode.commands.registerCommand('llamacpp.switchModel', switchModel)
    );

    checkCopilotConflict(context);
}

async function checkCopilotConflict(context: vscode.ExtensionContext): Promise<void> {
    const flagKey = 'copilotInlineDisabled';
    if (context.globalState.get<boolean>(flagKey)) { return; }

    const copilotEnabled = vscode.workspace
        .getConfiguration('github.copilot.editor')
        .get<boolean>('enableAutoCompletions', true);

    if (!copilotEnabled) {
        context.globalState.update(flagKey, true);
        return;
    }

    const choice = await vscode.window.showInformationMessage(
        'llama.cpp Local AI: Copilot inline completions are active and will compete with local completions. Disable Copilot ghost-text so llama.cpp handles all inline suggestions?',
        'Disable Copilot ghost-text',
        'Keep both',
    );

    if (choice === 'Disable Copilot ghost-text') {
        await vscode.workspace.getConfiguration('github.copilot.editor')
            .update('enableAutoCompletions', false, vscode.ConfigurationTarget.Global);
        out.appendLine('[setup] Copilot inline completions disabled — llama.cpp now handles all ghost-text');
    }

    context.globalState.update(flagKey, true);
}

async function switchModel(): Promise<void> {
    const ep = vscode.workspace.getConfiguration('llamacpp').get<string>('endpoint', 'http://localhost:18080');

    let items: { label: string; description: string }[] = [];
    try {
        const res = await fetch(`${ep}/v1/models`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            const json: any = await res.json();
            items = (json.data ?? []).map((m: { id: string }) => ({
                label: m.id,
                description: 'loaded on server',
            }));
        }
    } catch { /* server unreachable — fall through to static list */ }

    if (items.length === 0) {
        items = MODELS.map(m => ({ label: m.id, description: m.label }));
    }

    const pick = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select model for @llama chat and inline completions',
    });
    if (!pick) { return; }
    await vscode.workspace.getConfiguration('llamacpp').update('model', pick.label, vscode.ConfigurationTarget.Global);
    vscode.window.showInformationMessage(`llama.cpp: model set to ${pick.label}`);
}

export function deactivate(): void {}
