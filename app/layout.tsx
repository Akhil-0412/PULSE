
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Navbar from './components/Navbar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'PulseGuard - AI Vital Monitoring',
  description: 'Showcasing ResNet-1D PPG Analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen bg-[#020617] text-slate-100 selection:bg-teal-500/30`}>
        <Navbar />
        <main className="pt-24 px-4 pb-12 w-full max-w-6xl mx-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
