
import { ArrowRight, Cpu, Filter, Zap, BrainCircuit, Activity } from 'lucide-react';

export default function ProcessPage() {
    return (
        <div className="space-y-12 animate-in slide-in-from-bottom-4 duration-700">

            <div className="space-y-4">
                <h1 className="text-4xl font-bold text-white">The Evolution</h1>
                <p className="text-slate-400 max-w-2xl">
                    How we pivoted from a complex Transformer to a streamlined ResNet-1D to achieve medical-grade accuracy.
                </p>
            </div>

            {/* Timeline Steps */}
            <div className="relative border-l border-slate-800 ml-4 space-y-12 pb-12">

                {/* Phase 1 */}
                <div className="pl-10 relative">
                    <div className="absolute -left-3 top-0 w-6 h-6 rounded-full bg-slate-800 border-4 border-[#020617] ring-1 ring-slate-700" />

                    <div className="glass-card p-6 border-slate-700/50">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 bg-purple-500/10 rounded text-purple-400">
                                <Cpu className="w-5 h-5" />
                            </div>
                            <h2 className="text-xl font-bold text-slate-200">Phase 1: The Transformer</h2>
                        </div>
                        <p className="text-slate-400 mb-4">
                            We started with a <code>TransPPG</code> architecture, assuming Self-Attention was needed
                            to model long-range dependencies in the 8-second window.
                        </p>
                        <div className="grid grid-cols-2 gap-4 bg-slate-900/50 p-4 rounded-lg border border-slate-800">
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-semibold">MAE Score</div>
                                <div className="text-2xl font-bold text-red-400">11.11 BPM</div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-semibold">Issue</div>
                                <div className="text-sm text-slate-300">Overfitting to noise</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Pivot Point */}
                <div className="pl-10 relative">
                    <div className="absolute -left-3 top-0 w-6 h-6 rounded-full bg-teal-500 border-4 border-[#020617] ring-1 ring-teal-500 shadow-[0_0_10px_rgba(20,184,166,0.5)]" />

                    <div className="glass-card p-6 border-teal-500/30 bg-teal-900/10">
                        <h2 className="text-xl font-bold text-teal-300 mb-2">The Optimization</h2>
                        <p className="text-teal-100/70 mb-4">
                            We analyzed the failure cases and realized the model wasn't lacking *capacity*,
                            it was lacking *clarity*.
                        </p>
                        <div className="flex flex-wrap gap-2">
                            <span className="px-3 py-1 rounded-full bg-teal-500/20 text-teal-300 text-xs font-semibold border border-teal-500/20 flex items-center gap-1">
                                <Filter className="w-3 h-3" /> 0.5-8Hz Filter
                            </span>
                            <span className="px-3 py-1 rounded-full bg-teal-500/20 text-teal-300 text-xs font-semibold border border-teal-500/20 flex items-center gap-1">
                                <Zap className="w-3 h-3" /> 16s Window
                            </span>
                        </div>
                    </div>
                </div>

                {/* Phase 2 */}
                <div className="pl-10 relative">
                    <div className="absolute -left-3 top-0 w-6 h-6 rounded-full bg-slate-800 border-4 border-[#020617] ring-1 ring-slate-700" />

                    <div className="glass-card p-6 border-slate-700/50">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 bg-blue-500/10 rounded text-blue-400">
                                <Activity className="w-5 h-5" />
                            </div>
                            <h2 className="text-xl font-bold text-slate-200">Phase 2: ResNet-1D</h2>
                        </div>
                        <p className="text-slate-400 mb-4">
                            We switched to a lighter, deeper Convolutional Network. The 1D-CNN was extremely
                            efficient on the RTX 4060, allowing faster convergence.
                        </p>
                        <div className="grid grid-cols-2 gap-4 bg-slate-900/50 p-4 rounded-lg border border-slate-800">
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-semibold">MAE Score</div>
                                <div className="text-2xl font-bold text-green-400">5.93 BPM</div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-semibold">Improvement</div>
                                <div className="text-sm text-green-300 font-bold">47% Reduction</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Phase 3 */}
                <div className="pl-10 relative">
                    <div className="absolute -left-3 top-0 w-6 h-6 rounded-full bg-rose-500 border-4 border-[#020617] ring-1 ring-rose-500 shadow-[0_0_15px_rgba(244,63,94,0.6)]" />

                    <div className="glass-card p-6 border-rose-500/30 bg-rose-900/10">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 bg-rose-500/10 rounded text-rose-400">
                                <BrainCircuit className="w-5 h-5" />
                            </div>
                            <h2 className="text-xl font-bold text-white">Phase 3: AttentionCNNLSTM <span className="text-xs bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded ml-2 border border-rose-500/30">CUSTOM</span></h2>
                        </div>
                        <p className="text-rose-100/80 mb-4">
                            To break the 5.9 BPM barrier, we engineered a <strong>custom hybrid architecture</strong>.
                            We combined the feature extraction power of CNNs with the temporal memory of Bi-LSTMs and
                            added an Attention mechanism to focus on the cleanest signal segments.
                        </p>
                        <div className="grid grid-cols-2 gap-4 bg-slate-900/50 p-4 rounded-lg border border-slate-800">
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-semibold">MAE Score</div>
                                <div className="text-2xl font-bold text-white">5.40 BPM</div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-500 uppercase font-semibold">Status</div>
                                <div className="text-sm text-green-400 font-bold flex items-center gap-2">
                                    Best Model Yet <Zap className="w-3 h-3" />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
