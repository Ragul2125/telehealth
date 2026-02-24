import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { format } from 'date-fns';
import { Activity, Droplets, Heart, Thermometer, TrendingUp } from 'lucide-react';
import clsx from 'clsx';

// Chart configurations
const CHARTS = [
    {
        key: 'heartRate',
        label: 'Heart Rate',
        unit: 'bpm',
        color: '#6366f1',
        refHigh: 100,
        refLow: 60,
        Icon: Heart,
        yDomain: [40, 180],
    },
    {
        key: 'spo2',
        label: 'SpO₂',
        unit: '%',
        color: '#10b981',
        refLow: 95,
        Icon: Droplets,
        yDomain: [80, 100],
    },
    {
        key: 'systolicBP',
        label: 'Systolic BP',
        unit: 'mmHg',
        color: '#f59e0b',
        refHigh: 140,
        refLow: 90,
        Icon: Activity,
        yDomain: [60, 200],
    },
    {
        key: 'riskScore',
        label: 'Risk Score',
        unit: '',
        color: '#ef4444',
        refHigh: 0.6,
        Icon: TrendingUp,
        yDomain: [0, 1],
    },
];

const CustomTooltip = ({ active, payload, label, unit }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-white dark:bg-gray-800 shadow-lg rounded-xl border border-gray-100 dark:border-gray-700 px-3 py-2 text-xs">
            <p className="text-gray-500 dark:text-gray-400 mb-1">{label}</p>
            {payload.map((p) => (
                <p key={p.dataKey} className="font-semibold" style={{ color: p.color }}>
                    {p.name}: {typeof p.value === 'number' || !isNaN(Number(p.value)) ? Number(p.value).toFixed(1) : p.value}
                    {unit && ` ${unit}`}
                </p>
            ))}
        </div>
    );
};

function SingleChart({ chart, data }) {
    const { Icon } = chart;
    const chartData = data.map((d) => ({
        ...d,
        time: format(new Date(d.timestamp), 'HH:mm'),
        value: Number(d[chart.key]),
    }));

    return (
        <div className="card">
            <div className="flex items-center gap-2 mb-4">
                <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: chart.color + '20' }}
                >
                    <Icon className="w-4 h-4" style={{ color: chart.color }} />
                </div>
                <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
                        {chart.label}
                    </h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                        Last {data.length} readings
                    </p>
                </div>
                <div className="ml-auto text-right">
                    <span className="text-xl font-bold" style={{ color: chart.color }}>
                        {chartData.length > 0 && !isNaN(Number(chartData.at(-1)?.value))
                            ? Number(chartData.at(-1).value).toFixed(1)
                            : '--'}
                    </span>
                    {chart.unit && (
                        <span className="text-xs text-gray-400 ml-1">{chart.unit}</span>
                    )}
                </div>
            </div>

            <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" className="dark:stroke-gray-800" />
                    <XAxis
                        dataKey="time"
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                        tickLine={false}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        domain={chart.yDomain}
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                        tickLine={false}
                        axisLine={false}
                    />
                    <Tooltip
                        content={<CustomTooltip unit={chart.unit} />}
                        cursor={{ stroke: chart.color, strokeWidth: 1, strokeDasharray: '4 4' }}
                    />
                    {chart.refHigh && (
                        <ReferenceLine y={chart.refHigh} stroke={chart.color} strokeDasharray="4 4" strokeOpacity={0.5} />
                    )}
                    {chart.refLow && (
                        <ReferenceLine y={chart.refLow} stroke={chart.color} strokeDasharray="4 4" strokeOpacity={0.5} />
                    )}
                    <Line
                        type="monotone"
                        dataKey="value"
                        stroke={chart.color}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: chart.color, strokeWidth: 2, stroke: '#fff' }}
                        name={chart.label}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

/**
 * VitalsChart — renders 4 Recharts line charts in a 2×2 responsive grid.
 * @param {Array} data - vitalsHistory array from PatientContext
 */
export default function VitalsChart({ data = [] }) {
    if (!data.length) {
        return (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {CHARTS.map((chart) => (
                    <div key={chart.key} className="card animate-pulse h-56 bg-gray-50 dark:bg-gray-800" />
                ))}
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {CHARTS.map((chart) => (
                <SingleChart key={chart.key} chart={chart} data={data} />
            ))}
        </div>
    );
}
