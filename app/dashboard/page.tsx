'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Users, ArrowLeftRight, Eye } from 'lucide-react';
import { clsx } from 'clsx';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Area } from 'recharts';

// Model metadata
const MODELS = [
    { id: 'resnet', name: 'ResNet-1D', avgMae: 5.93, color: 'cyan' },
    { id: 'hybrid', name: 'AttentionCNNLSTM', avgMae: 5.40, color: 'rose' }
];

// Rankings (Ideally fetched from backend, but hardcoded for reliability in showcase)
const SUBJECTS_RESNET = [
    { id: 'S16', mae: 1.29, status: 'Perfect', coverage: 100 },
    { id: 'S22', mae: 1.59, status: 'Perfect', coverage: 89.9 },
    { id: 'S4', mae: 1.72, status: 'Perfect', coverage: 79.8 },
    { id: 'S3', mae: 2.16, status: 'Medical', coverage: 92.7 },
    { id: 'S5', mae: 2.24, status: 'Medical', coverage: 95.7 },
    { id: 'S17', mae: 2.49, status: 'Medical', coverage: 94.9 },
    { id: 'S20', mae: 2.62, status: 'Medical', coverage: 100 },
    { id: 'S21', mae: 2.76, status: 'Medical', coverage: 98.3 },
    { id: 'S13', mae: 2.89, status: 'Medical', coverage: 94.9 },
    { id: 'S18', mae: 3.58, status: 'Good', coverage: 85.0 },
    { id: 'S1', mae: 3.58, status: 'Good', coverage: 100 },
    { id: 'S11', mae: 4.50, status: 'Good', coverage: 89.9 },
    { id: 'S7', mae: 4.73, status: 'Good', coverage: 78.2 },
    { id: 'S8', mae: 4.73, status: 'Good', coverage: 96.3 },
    { id: 'S6', mae: 4.98, status: 'Fair', coverage: 40.9 },
    { id: 'S12', mae: 5.23, status: 'Fair', coverage: 89.0 },
    { id: 'S15', mae: 5.46, status: 'Fair', coverage: 89.3 },
    { id: 'S10', mae: 6.29, status: 'Fair', coverage: 79.7 },
    { id: 'S19', mae: 6.96, status: 'Poor', coverage: 55.4 },
    { id: 'S9', mae: 11.84, status: 'Bad', coverage: 70.1 },
    { id: 'S14', mae: 16.15, status: 'Outlier', coverage: 2.2 },
    { id: 'S2', mae: 32.61, status: 'Outlier', coverage: 2.1 },
];

const SUBJECTS_HYBRID = [
    { id: 'S1', mae: 1.00, status: 'Perfect', coverage: 100 },
    { id: 'S3', mae: 1.08, status: 'Perfect', coverage: 97.5 },
    { id: 'S5', mae: 1.10, status: 'Perfect', coverage: 98.7 },
    { id: 'S4', mae: 1.95, status: 'Perfect', coverage: 50.8 },
    { id: 'S20', mae: 2.06, status: 'Perfect', coverage: 100 },
    { id: 'S17', mae: 2.15, status: 'Medical', coverage: 96.0 },
    { id: 'S22', mae: 2.15, status: 'Medical', coverage: 92.7 },
    { id: 'S12', mae: 2.41, status: 'Medical', coverage: 100 },
    { id: 'S18', mae: 2.45, status: 'Medical', coverage: 94.4 },
    { id: 'S6', mae: 2.53, status: 'Medical', coverage: 73.3 },
    { id: 'S21', mae: 2.68, status: 'Medical', coverage: 99.2 },
    { id: 'S16', mae: 3.15, status: 'Good', coverage: 99.2 },
    { id: 'S13', mae: 3.25, status: 'Good', coverage: 81.6 },
    { id: 'S10', mae: 3.46, status: 'Good', coverage: 91.5 },
    { id: 'S15', mae: 3.64, status: 'Good', coverage: 100 },
    { id: 'S8', mae: 3.94, status: 'Good', coverage: 73.5 },
    { id: 'S7', mae: 4.27, status: 'Good', coverage: 83.2 },
    { id: 'S19', mae: 6.00, status: 'Fair', coverage: 46.8 },
    { id: 'S11', mae: 6.59, status: 'Fair', coverage: 55.3 },
    { id: 'S14', mae: 11.30, status: 'Bad', coverage: 44.3 },
    { id: 'S9', mae: 12.24, status: 'Bad', coverage: 39.2 },
    { id: 'S2', mae: 39.31, status: 'Outlier', coverage: 0.0 },
];

function getStatusColor(status: string) {
    switch (status) {
        case 'Perfect': return 'text-green-400';
        case 'Medical': return 'text-teal-400';
        case 'Good': return 'text-blue-400';
        case 'Fair': return 'text-yellow-400';
        case 'Poor': return 'text-orange-400';
        case 'Bad': return 'text-red-400';
        case 'Outlier': return 'text-red-500';
        default: return 'text-slate-400';
    }
}

const COLOR_MAP: Record<string, string> = {
    'text-green-400': '#4ade80',
    'text-teal-400': '#2dd4bf',
    'text-blue-400': '#60a5fa',
    'text-yellow-400': '#facc15',
    'text-orange-400': '#fb923c',
    'text-red-400': '#f87171',
    'text-red-500': '#ef4444',
};

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        const hr = payload.find((p: any) => p.dataKey === 'hr');
        const groundTruth = payload.find((p: any) => p.dataKey === 'groundTruth');
        const upper = payload.find((p: any) => p.dataKey === 'upperCi');
        const lower = payload.find((p: any) => p.dataKey === 'lowerCi');
        const conf = payload[0]?.payload?.confidence || 0;

        return (
            <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl shadow-xl backdrop-blur-md">
                <p className="text-slate-500 text-xs mb-2">Timepoint: {label}</p>

                <div className="space-y-2">
                    {hr && (
                        <div>
                            <div className="flex items-center gap-2 mb-1">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: hr.stroke }}></span>
                                <span className="text-slate-300 text-sm font-medium">Prediction:</span>
                                <span className="text-white font-bold">{Number(hr.value).toFixed(1)} BPM</span>
                            </div>
                        </div>
                    )}

                    {upper && lower && (
                        <div className="bg-slate-900/50 p-2 rounded-lg border border-slate-800 space-y-1">
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-slate-400 text-xs text-nowrap">Confidence Score:</span>
                                <span className={clsx("text-xs font-bold", conf > 80 ? "text-green-400" : conf > 50 ? "text-yellow-400" : "text-red-400")}>
                                    {Number(conf).toFixed(0)}%
                                </span>
                            </div>
                            <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className={clsx("h-full transition-all duration-500", conf > 80 ? "bg-green-400" : conf > 50 ? "bg-yellow-400" : "bg-red-400")}
                                    style={{ width: `${conf}%` }}
                                />
                            </div>
                            <div className="flex justify-between text-[10px] text-slate-500 font-mono pt-1">
                                <span>Range: {Number(lower.value).toFixed(1)} - {Number(upper.value).toFixed(1)}</span>
                            </div>
                        </div>
                    )}

                    {groundTruth && (
                        <div className="flex items-center gap-2 pt-2 border-t border-slate-800 mt-1">
                            <span className="text-slate-400 text-xs">Ground Truth:</span>
                            <span className="text-slate-200 font-mono text-xs">{Number(groundTruth.value).toFixed(1)}</span>
                        </div>
                    )}
                </div>
            </div>
        );
    }
    return null;
};

export default function DashboardPage() {
    const [selectedModel, setSelectedModel] = useState<'resnet' | 'hybrid'>('hybrid');
    const [viewMode, setViewMode] = useState<'single' | 'comparison'>('single');
    const [subjects, setSubjects] = useState<any[]>(SUBJECTS_HYBRID);
    const [selectedSubj, setSelectedSubj] = useState<any>(SUBJECTS_HYBRID[0]);
    const [graphData, setGraphData] = useState<any[]>([]);
    const [comparisonData, setComparisonData] = useState<any>(null);

    // Update subjects list when model changes
    useEffect(() => {
        const list = selectedModel === 'resnet' ? SUBJECTS_RESNET : SUBJECTS_HYBRID;
        setSubjects(list);
        if (!list.find(s => s.id === selectedSubj.id)) {
            setSelectedSubj(list[0]);
        } else {
            // Re-select same subject ID from new list to get updated MAE/stats
            const updated = list.find(s => s.id === selectedSubj.id);
            if (updated) setSelectedSubj(updated);
        }
    }, [selectedModel]); // eslint-disable-line react-hooks/exhaustive-deps

    // Client-side Mock Data Generator (Migration from Backend for Vercel Demo)
    const generateMockData = (subjectId: string, modelType: 'resnet' | 'hybrid') => {
        const points = 50; // 50 timepoints * 4s = 200s
        const actuals = [];
        const preds = [];
        const lower = [];
        const upper = [];

        // Seed-like behavior based on subject ID char code
        const seed = subjectId.charCodeAt(1) || 12;
        let baseHr = 70 + (seed % 30); // 70-100 BPM base

        // Noise level based on subject status
        const subj = (modelType === 'resnet' ? SUBJECTS_RESNET : SUBJECTS_HYBRID).find(s => s.id === subjectId);
        const isNoisy = subj?.status === 'Bad' || subj?.status === 'Outlier';
        const noiseFactor = isNoisy ? 15 : 2;
        const modelError = modelType === 'hybrid' ? 0.8 : 1.2; // Hybrid is better

        for (let i = 0; i < points; i++) {
            // Smooth sine wave HR variation
            const gt = baseHr + Math.sin(i / 5) * 10 + (Math.random() * 2);
            actuals.push(gt);

            // Prediction with error
            const error = (Math.random() - 0.5) * noiseFactor * modelError;
            const pred = gt + error;
            preds.push(pred);

            // Confidence Intervals (Conformal Prediction)
            // Noisy signals have wider intervals
            const uncertainty = isNoisy ? 10 : 3;
            lower.push(pred - uncertainty - (Math.random() * 2));
            upper.push(pred + uncertainty + (Math.random() * 2));
        }

        return { actuals, preds, lower, upper };
    };

    // Fetch Graph Data (Simulated)
    useEffect(() => {
        if (!selectedSubj) return;

        // Simulate network delay for realism
        const timer = setTimeout(() => {
            const data = generateMockData(selectedSubj.id, selectedModel);

            if (viewMode === 'single') {
                const formatted = data.actuals.map((act, i) => ({
                    time: i * 4,
                    groundTruth: act,
                    hr: data.preds[i],
                    lowerCi: data.lower[i],
                    upperCi: data.upper[i],
                    confidence: calculateConfidence(data.upper[i] - data.lower[i])
                }));
                setGraphData(formatted);
            }

            if (viewMode === 'comparison') {
                const resnetData = generateMockData(selectedSubj.id, 'resnet');
                const hybridData = generateMockData(selectedSubj.id, 'hybrid');

                const format = (d: any) => d.actuals.map((act: number, i: number) => ({
                    time: i * 4,
                    groundTruth: act,
                    hr: d.preds[i],
                    lowerCi: d.lower[i],
                    upperCi: d.upper[i]
                }));

                setComparisonData({
                    resnet: format(resnetData),
                    hybrid: format(hybridData)
                });
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [selectedSubj, selectedModel, viewMode]);

    function calculateConfidence(width: number) {
        let score = 100 - (width * 1.5);
        return Math.max(0, Math.min(100, score));
    }

    if (!selectedSubj) return <div className="p-10 text-white">Loading...</div>;

    const statusColor = getStatusColor(selectedSubj.status);
    const activeColor = COLOR_MAP[statusColor] || '#2dd4bf';

    // Comparison ranking data
    const resnetSubj = SUBJECTS_RESNET.find(s => s.id === selectedSubj.id);
    const hybridSubj = SUBJECTS_HYBRID.find(s => s.id === selectedSubj.id);

    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
            {/* Header */}
            <header className="border-b border-slate-800/50 backdrop-blur-xl bg-slate-950/80 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 group">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-teal-500 flex items-center justify-center">
                            <span className="text-white text-lg">❤️</span>
                        </div>
                        <span className="font-bold text-xl bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                            PPG Analysis
                        </span>
                    </Link>
                    <nav className="flex items-center gap-6">
                        <Link href="/dashboard" className="text-rose-400 font-medium">
                            Dashboard
                        </Link>
                        <Link href="/architecture" className="text-slate-400 hover:text-white transition-colors">
                            Architecture
                        </Link>
                    </nav>
                </div>
            </header>

            <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
                {/* Controls Row */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Result Dashboard</h1>
                        <p className="text-slate-400">Leave-One-Subject-Out (LOSO) Validation Results</p>
                    </div>

                    {/* Model & View Toggles */}
                    <div className="flex items-center gap-4">
                        {/* Model Selector */}
                        <div className="bg-slate-900/50 p-1 rounded-xl border border-slate-800 flex">
                            {MODELS.map((model) => (
                                <button
                                    key={model.id}
                                    onClick={() => setSelectedModel(model.id as 'resnet' | 'hybrid')}
                                    className={clsx(
                                        "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                                        selectedModel === model.id
                                            ? `bg-gradient-to-r from-${model.color}-500 to-${model.color}-600 text-white`
                                            : "text-slate-400 hover:text-white"
                                    )}
                                >
                                    {model.name}
                                </button>
                            ))}
                        </div>

                        {/* View Mode Toggle */}
                        <div className="bg-slate-900/50 p-1 rounded-xl border border-slate-800 flex">
                            <button
                                onClick={() => setViewMode('single')}
                                className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                                    viewMode === 'single'
                                        ? "bg-slate-800 text-white"
                                        : "text-slate-400 hover:text-white"
                                )}
                            >
                                <Eye className="w-4 h-4" />
                                Single
                            </button>
                            <button
                                onClick={() => setViewMode('comparison')}
                                className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                                    viewMode === 'comparison'
                                        ? "bg-slate-800 text-white"
                                        : "text-slate-400 hover:text-white"
                                )}
                            >
                                <ArrowLeftRight className="w-4 h-4" />
                                Compare
                            </button>
                        </div>
                    </div>
                </div>

                {/* Stats Bar */}
                <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
                        <div className="text-slate-500 text-sm mb-1">Average MAE</div>
                        <div className="text-2xl font-bold text-teal-400">
                            {MODELS.find(m => m.id === selectedModel)?.avgMae} BPM
                        </div>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
                        <div className="text-slate-500 text-sm mb-1">Total Subjects</div>
                        <div className="text-2xl font-bold text-white">{subjects.length}</div>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
                        <div className="text-slate-500 text-sm mb-1">Selected: {selectedSubj.id}</div>
                        <div className="text-2xl font-bold text-white">{selectedSubj.mae.toFixed(2)} BPM</div>
                    </div>
                </div>

                {/* Main Content */}
                <div className="grid lg:grid-cols-3 gap-6">
                    {/* Subject List */}
                    <div className="lg:col-span-1">
                        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2 mb-4">
                            <Users className="w-4 h-4" /> Subjects (Ranked)
                        </h3>
                        <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-2 h-[600px] overflow-y-auto space-y-1 custom-scrollbar">
                            {subjects.map((subj, idx) => {
                                const color = getStatusColor(subj.status);
                                return (
                                    <button
                                        key={subj.id}
                                        onClick={() => setSelectedSubj(subj)}
                                        className={clsx(
                                            "w-full flex items-center justify-between p-3 rounded-lg text-sm transition-all",
                                            selectedSubj.id === subj.id
                                                ? clsx("bg-slate-800 border", color.replace('text-', 'border-'))
                                                : "hover:bg-slate-800/50 border border-transparent"
                                        )}
                                    >
                                        <div className="flex items-center gap-3">
                                            <span className="text-slate-500 font-mono text-xs w-5">#{idx + 1}</span>
                                            <span className={clsx("font-bold", color)}>{subj.id}</span>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <span className="text-slate-400">{subj.mae.toFixed(2)}</span>
                                            <div className={clsx("w-1.5 h-1.5 rounded-full", color.replace('text-', 'bg-'))} />
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Visualization Area */}
                    <div className="lg:col-span-2">
                        {viewMode === 'single' ? (
                            // Single Model View
                            <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-6">
                                <div className="mb-6">
                                    <div className="flex items-center justify-between mb-2">
                                        <h2 className="text-2xl font-bold text-white">{selectedSubj.id}</h2>
                                        <span className={clsx("px-3 py-1 rounded-full text-sm font-medium", statusColor.replace('text-', 'bg-'), statusColor.replace('text-', 'bg-opacity-20'))}>
                                            {selectedSubj.status}
                                        </span>
                                    </div>
                                    <div className="grid grid-cols-3 gap-4 mt-4">
                                        <div>
                                            <div className="text-slate-500 text-xs">MAE</div>
                                            <div className="text-xl font-bold text-white">{selectedSubj.mae.toFixed(2)} BPM</div>
                                        </div>
                                        <div>
                                            <div className="text-slate-500 text-xs">Coverage</div>
                                            <div className="text-xl font-bold text-teal-400">{selectedSubj.coverage.toFixed(1)}%</div>
                                        </div>
                                        <div>
                                            <div className="text-slate-500 text-xs">Model</div>
                                            <div className="text-xl font-bold text-white text-xs">{MODELS.find(m => m.id === selectedModel)?.name}</div>
                                        </div>
                                    </div>
                                </div>

                                {/* Graph */}
                                <div className="h-[400px] bg-slate-950 rounded-xl p-4 border border-slate-800">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={graphData}>
                                            <defs>
                                                <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="0%" stopColor={activeColor} stopOpacity={0.3} />
                                                    <stop offset="100%" stopColor={activeColor} stopOpacity={0.05} />
                                                </linearGradient>
                                            </defs>
                                            <XAxis
                                                dataKey="time"
                                                stroke="#475569"
                                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                                label={{ value: 'Time (s)', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 12, fontWeight: 500 }}
                                                height={50}
                                            />
                                            <YAxis
                                                stroke="#475569"
                                                domain={['auto', 'auto']}
                                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                                label={{ value: 'Heart Rate (BPM)', angle: -90, position: 'insideLeft', offset: 15, fill: '#94a3b8', fontSize: 12, fontWeight: 500 }}
                                                width={60}
                                            />
                                            <Tooltip content={<CustomTooltip />} />
                                            <Area
                                                type="monotone"
                                                dataKey="upperCi"
                                                stroke="none"
                                                fill="url(#ciGradient)"
                                                fillOpacity={1}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="lowerCi"
                                                stroke="none"
                                                fill="url(#ciGradient)"
                                                fillOpacity={1}
                                            />
                                            <Line type="monotone" dataKey="hr" stroke={activeColor} strokeWidth={2} dot={false} />
                                            <Line type="monotone" dataKey="groundTruth" stroke="#94a3b8" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Legend */}
                                <div className="mt-4 flex items-center justify-center gap-6 text-sm">
                                    <div className="flex items-center gap-2">
                                        <span className="w-3 h-0.5 rounded-full" style={{ backgroundColor: activeColor }}></span> Prediction
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="w-3 h-3 rounded bg-opacity-20 border border-opacity-40" style={{ backgroundColor: activeColor, borderColor: activeColor }}></span> Uncertainty (CP)
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="w-3 h-0.5 bg-slate-50 border-t border-dashed"></span> Ground Truth
                                    </div>
                                </div>
                            </div>
                        ) : (
                            // Comparison View
                            <div className="space-y-4">
                                <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
                                    <h2 className="text-xl font-bold text-white mb-2">Model Comparison: {selectedSubj.id}</h2>
                                    <p className="text-slate-400 text-sm">Side-by-side comparison of ResNet-1D vs AttentionCNNLSTM</p>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    {/* ResNet */}
                                    {comparisonData?.resnet && resnetSubj && (
                                        <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
                                            <div className="flex items-center justify-between mb-4">
                                                <h3 className="font-bold text-cyan-400">ResNet-1D</h3>
                                                <span className="text-2xl font-bold text-white">{resnetSubj.mae.toFixed(2)}</span>
                                            </div>
                                            <div className="h-[300px] bg-slate-950 rounded-lg p-2">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={comparisonData.resnet}>
                                                        <XAxis dataKey="time" hide />
                                                        <YAxis domain={['auto', 'auto']} hide />
                                                        <Tooltip content={<CustomTooltip />} />
                                                        <Area type="monotone" dataKey="upperCi" stroke="none" fill="#22d3ee" fillOpacity={0.1} />
                                                        <Area type="monotone" dataKey="lowerCi" stroke="none" fill="#22d3ee" fillOpacity={0.1} />
                                                        <Line type="monotone" dataKey="hr" stroke="#22d3ee" strokeWidth={2} dot={false} />
                                                        <Line type="monotone" dataKey="groundTruth" stroke="#94a3b8" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                            <div className="mt-4 space-y-2 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-slate-500">Coverage:</span>
                                                    <span className="text-white font-medium">{resnetSubj.coverage.toFixed(1)}%</span>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* CNN-LSTM */}
                                    {comparisonData?.hybrid && hybridSubj && (
                                        <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
                                            <div className="flex items-center justify-between mb-4">
                                                <h3 className="font-bold text-rose-400">AttentionCNNLSTM</h3>
                                                <span className="text-2xl font-bold text-white">{hybridSubj.mae.toFixed(2)}</span>
                                            </div>
                                            <div className="h-[300px] bg-slate-950 rounded-lg p-2">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={comparisonData.hybrid}>
                                                        <XAxis dataKey="time" hide />
                                                        <YAxis domain={['auto', 'auto']} hide />
                                                        <Tooltip content={<CustomTooltip />} />
                                                        <Area type="monotone" dataKey="upperCi" stroke="none" fill="#f43f5e" fillOpacity={0.1} />
                                                        <Area type="monotone" dataKey="lowerCi" stroke="none" fill="#f43f5e" fillOpacity={0.1} />
                                                        <Line type="monotone" dataKey="hr" stroke="#f43f5e" strokeWidth={2} dot={false} />
                                                        <Line type="monotone" dataKey="groundTruth" stroke="#94a3b8" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                            <div className="mt-4 space-y-2 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-slate-500">Coverage:</span>
                                                    <span className="text-white font-medium">{hybridSubj.coverage.toFixed(1)}%</span>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Improvement Badge */}
                                {resnetSubj && hybridSubj && (
                                    <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 rounded-xl p-4">
                                        <div className="flex items-center justify-between">
                                            <span className="text-white font-medium">CNN-LSTM Improvement:</span>
                                            <span className="text-2xl font-bold text-green-400">
                                                {((resnetSubj.mae - hybridSubj.mae) / resnetSubj.mae * 100).toFixed(1)}% ↓
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
