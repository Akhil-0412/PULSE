import { NextResponse } from 'next/server';
import { resultsData } from '@/lib/results-data';

export async function GET() {
    return NextResponse.json(resultsData, {
        headers: {
            'Cache-Control': 'public, max-age=31536000, immutable',
        },
    });
}
