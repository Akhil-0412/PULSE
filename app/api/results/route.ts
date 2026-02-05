import { NextResponse } from 'next/server';
import { resultsData } from '@/lib/results-data';

export async function GET() {
    return NextResponse.json(resultsData, {
        headers: {
            'Cache-Control': 'no-store',
        },
    });
}
