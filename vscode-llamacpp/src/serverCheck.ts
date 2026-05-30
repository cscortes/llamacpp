import * as vscode from 'vscode';
import { endpoint, activeModel } from './models';
import { out } from './extension';

let statusBar: vscode.StatusBarItem;

export function registerServerCheck(context: vscode.ExtensionContext): void {
    statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.command = 'llamacpp.changeEndpoint';
    context.subscriptions.push(statusBar);
    statusBar.show();

    context.subscriptions.push(
        vscode.commands.registerCommand('llamacpp.changeEndpoint', changeEndpoint),
        vscode.commands.registerCommand('llamacpp.checkServer', checkAndReport),
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('llamacpp')) { lastOnline = undefined; updateStatus(); }
        })
    );

    updateStatus();
    setInterval(updateStatus, 30_000);
}

async function changeEndpoint(): Promise<void> {
    const current = endpoint();
    const value = await vscode.window.showInputBox({
        title: 'llama.cpp Server Endpoint',
        prompt: 'Enter the base URL of your running llama.cpp server',
        value: current,
        valueSelection: [0, current.length],
        validateInput: input => {
            try { new URL(input); return undefined; }
            catch { return 'Must be a valid URL, e.g. http://172.26.156.205:18080'; }
        },
    });
    if (value === undefined || value === current) { return; }
    await vscode.workspace.getConfiguration('llamacpp').update(
        'endpoint', value, vscode.ConfigurationTarget.Global
    );
    out.appendLine(`[${ts()}] Endpoint changed to ${value}`);
}

async function checkAndReport(): Promise<void> {
    const ok = await ping();
    const msg = ok
        ? `llama.cpp server is reachable at ${endpoint()}`
        : `llama.cpp server not found at ${endpoint()} — is the container running?`;
    out.appendLine(`[${ts()}] Manual check: ${ok ? 'PASS' : 'FAIL'} — ${endpoint()}`);
    out.show(true);
    vscode.window.showInformationMessage(msg);
}

let lastOnline: boolean | undefined;

async function updateStatus(): Promise<void> {
    const model = activeModel();
    const online = await ping();

    if (online !== lastOnline) {
        lastOnline = online;
        if (online) {
            out.appendLine(`[${ts()}] Server online — ${endpoint()}  model: ${model.id}`);
        } else {
            out.appendLine(`[${ts()}] Server offline — ${endpoint()} unreachable`);
        }
    }

    if (online) {
        statusBar.text = `$(sparkle) llama: ${model.id}`;
        statusBar.tooltip = `${model.label}\n${model.detail}\n${endpoint()}`;
        statusBar.backgroundColor = undefined;
    } else {
        statusBar.text = `$(warning) llama: offline`;
        statusBar.tooltip = `Server not reachable at ${endpoint()}\nRun: make server`;
        statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    }
}

function ts(): string {
    return new Date().toLocaleTimeString();
}

export async function ping(): Promise<boolean> {
    try {
        const res = await fetch(`${endpoint()}/v1/models`, { signal: AbortSignal.timeout(3000) });
        return res.ok;
    } catch {
        return false;
    }
}
