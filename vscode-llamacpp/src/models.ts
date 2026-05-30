import * as vscode from 'vscode';

export interface ModelDef {
    id: string;
    label: string;
    detail: string;
    supportsFim: boolean;
}

export const MODELS: ModelDef[] = [
    {
        id: 'phi',
        label: 'Phi-3.5-mini-instruct',
        detail: '2.5 GB · fastest · CPU-friendly',
        supportsFim: false,
    },
    {
        id: 'qwen2',
        label: 'Qwen2.5-Coder-7B-Instruct',
        detail: '4.4 GB · default · strong all-language coding',
        supportsFim: true,
    },
    {
        id: 'deep',
        label: 'DeepSeek-Coder-V2-Lite-Instruct',
        detail: '7.6 GB · most capable · best for complex tasks',
        supportsFim: true,
    },
];

export function getModelDef(id: string): ModelDef {
    return MODELS.find(m => m.id === id) ?? MODELS[1];
}

export function activeModel(): ModelDef {
    const id = vscode.workspace.getConfiguration('llamacpp').get<string>('model', 'qwen2');
    return getModelDef(id);
}

export function endpoint(): string {
    return vscode.workspace.getConfiguration('llamacpp').get<string>('endpoint', 'http://localhost:18080');
}
