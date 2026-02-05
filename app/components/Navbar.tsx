
'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { clsx } from 'clsx';
import { Activity, LayoutDashboard, FileText, Home } from 'lucide-react';

export default function Navbar() {
    const pathname = usePathname();

    const navItems = [
        { name: 'Overview', href: '/', icon: Home },
        { name: 'Process', href: '/process', icon: FileText },
        { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    ];

    return (
        <nav className="fixed top-0 w-full z-50 px-6 py-4">
            <div className="max-w-5xl mx-auto glass-panel rounded-2xl flex items-center justify-between px-6 py-3">

                {/* Logo */}
                <Link href="/" className="flex items-center gap-2 group">
                    <div className="p-2 bg-teal-500/10 rounded-lg group-hover:bg-teal-500/20 transition-colors">
                        <Activity className="w-6 h-6 text-teal-400" />
                    </div>
                    <span className="font-bold text-lg tracking-tight text-slate-100">
                        Pulse<span className="text-teal-400">Guard</span>
                    </span>
                </Link>

                {/* Links */}
                <div className="flex items-center gap-1">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href;
                        const Icon = item.icon;

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-all duration-200",
                                    isActive
                                        ? "bg-teal-500/10 text-teal-300 shadow-[0_0_10px_rgba(45,212,191,0.1)]"
                                        : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                                )}
                            >
                                <Icon className="w-4 h-4" />
                                {item.name}
                            </Link>
                        );
                    })}
                </div>
            </div>
        </nav>
    );
}
