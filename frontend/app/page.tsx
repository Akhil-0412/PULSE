
import Link from 'next/link';
import { ArrowRight, AlertTriangle, Activity, CheckCircle2 } from 'lucide-react';

export default function Home() {
  return (
    <div className="flex flex-col gap-16 animate-in fade-in duration-700">

      {/* Hero Section */}
      <section className="text-center space-y-6 pt-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-400 text-xs font-medium uppercase tracking-wider">
          <Activity className="w-4 h-4" /> ResNet-1D Enabled
        </div>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white">
          Precision Vitals from <br />
          <span className="text-gradient">Noisy Signals</span>
        </h1>

        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Recovering medical-grade heart rate data from motion-corrupted PPG sensors using
          advanced bandpass filtering and deep convolutional networks.
        </p>

        <div className="flex gap-4 justify-center pt-4">
          <Link href="/dashboard" className="px-6 py-3 bg-teal-500 hover:bg-teal-400 text-slate-900 font-bold rounded-xl transition-all flex items-center gap-2">
            View Dashboard <ArrowRight className="w-4 h-4" />
          </Link>
          <Link href="/process" className="px-6 py-3 glass-card hover:bg-slate-800 text-slate-200 font-semibold rounded-xl transition-all">
            Our Approach
          </Link>
        </div>
      </section>

      {/* The Problem Grid */}
      <div className="grid md:grid-cols-2 gap-8">

        {/* Card 1: The Issue */}
        <div className="glass-card p-8 space-y-4 border-red-500/20 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-32 bg-red-500/5 blur-3xl group-hover:bg-red-500/10 transition-all rounded-full" />

          <div className="w-12 h-12 rounded-lg bg-red-500/10 flex items-center justify-center text-red-500 mb-4">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h3 className="text-2xl font-bold text-slate-100">The Dataset Challenge</h3>
          <p className="text-slate-400 leading-relaxed">
            The PhysioNet Pulse Transit Time dataset contains severe artifacts.
            The README explicitly warns of <strong>"Issues in the files"</strong>, specifically
            sensor displacement during 'Run' activities.
          </p>
          <ul className="space-y-2 pt-2">
            <li className="flex items-center gap-2 text-red-300/80 text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> Signal Loss (0% Coverage in S2)
            </li>
            <li className="flex items-center gap-2 text-red-300/80 text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> Motion Artifacts &gt; 20 BPM Error
            </li>
          </ul>
        </div>

        {/* Card 2: The Solution */}
        <div className="glass-card p-8 space-y-4 border-teal-500/20 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-32 bg-teal-500/5 blur-3xl group-hover:bg-teal-500/10 transition-all rounded-full" />

          <div className="w-12 h-12 rounded-lg bg-teal-500/10 flex items-center justify-center text-teal-400 mb-4">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <h3 className="text-2xl font-bold text-slate-100">Our Analysis & Fix</h3>
          <p className="text-slate-400 leading-relaxed">
            We analyzed the signal frequency components and identified that the noise occupied
            specific bands. By applying a <strong>4th-Order Butterworth Filter (0.5-8Hz)</strong>,
            we effectively "cleaned" the input before it even reached the AI.
          </p>
          <ul className="space-y-2 pt-2">
            <li className="flex items-center gap-2 text-teal-300/80 text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-500" /> ResNet-1D Architecture
            </li>
            <li className="flex items-center gap-2 text-teal-300/80 text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-500" /> 16-Second Time Window
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
