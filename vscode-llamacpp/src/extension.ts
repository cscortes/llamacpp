import * as vscode from 'vscode';
import { MODELS, getModelDef } from './models';
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
}

async function switchModel(): Promise<void> {
    const items = MODELS.map(m => ({ label: m.id, description: m.label, detail: m.detail }));
    const pick = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select llama.cpp model (must match the model the server loaded)',
        matchOnDescription: true,
    });
    if (!pick) { return; }
    await vscode.workspace.getConfiguration('llamacpp').update('model', pick.label, vscode.ConfigurationTarget.Global);
    const def = getModelDef(pick.label);
    vscode.window.showInformationMessage(`llama.cpp: switched to ${def.label}`);
}

export function deactivate(): void {}
