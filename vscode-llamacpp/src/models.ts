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
        detail: '5.5 GB · Lite Q3 · 3060 stretch',
        supportsFim: true,
    },
    {
        id: 'qwen14',
        label: 'Qwen2.5-Coder-14B-Instruct',
        detail: '9 GB · 14B Q4 · 5060 Ti daily coding',
        supportsFim: true,
    },
    {
        id: 'deep5',
        label: 'DeepSeek-Coder-V2-Lite-Instruct Q5',
        detail: '12 GB · Lite Q5 · 5060 Ti coding stretch',
        supportsFim: true,
    },
    {
        id: 'chat3',
        label: 'Qwen2.5-3B-Instruct',
        detail: '2 GB · language · 3060',
        supportsFim: false,
    },
    {
        id: 'chat7',
        label: 'Qwen2.5-7B-Instruct',
        detail: '4.7 GB · language · 3060 / 5060 Ti',
        supportsFim: false,
    },
    {
        id: 'chat14',
        label: 'Qwen2.5-14B-Instruct',
        detail: '9 GB · 14B Q4 · 5060 Ti daily language',
        supportsFim: false,
    },
    {
        id: 'chat14q6',
        label: 'Qwen2.5-14B-Instruct Q6',
        detail: '12 GB · 14B Q6 · 5060 Ti language stretch',
        supportsFim: false,
    },
    {
        id: 'aya8',
        label: 'Aya Expanse 8B',
        detail: '5 GB · 23 languages · 3060 stretch / 5060 easy',
        supportsFim: false,
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
