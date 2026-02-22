import { useState, useCallback, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { usePatient } from '../context/PatientContext';
import Navbar from '../components/Navbar';
import VitalsChart from '../components/VitalsChart';
import AlertsTable from '../components/AlertsTable';
import RiskBadge from '../components/RiskBadge';
import { getBriefing } from '../api/briefingApi';
import {
    Search, RefreshCw, User, Loader2, Activity,
    FileText, Zap, Brain,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';
import clsx from 'clsx';

// Debounce hook
function useDebounce(value, delay) {
    const [deb, setDeb] = useState(value);
    useEffect(() => {
        const t = setTimeout(() => setDeb(value), delay);
        return () => clearTimeout(t);
    }, [value, delay]);
    return deb;
}

// Stat card
function StatCard({ label, value, unit, sub, color = 'indigo', icon: Icon }) {
    return (
        <div className="card flex flex-col gap-1">
            <div className="flex items-center justify-between">
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">{label}</p>
                {Icon && (
                    <div className={`w-7 h-7 rounded-lg bg-${color}-50 dark:bg-${color}-900/30 flex items-center justify-center`}>
                        <Icon className={`w-3.5 h-3.5 text-${color}-500`} />
                    </div>
                )}
            </div>
            <p className={`text-3xl font-bold text-${color}-600 dark:text-${color}-400 leading-none mt-1`}>
                {value ?? '—'}
                {unit && <span className="text-base font-normal text-gray-400 ml-1">{unit}</span>}
            </p>
            {sub && <p className="text-xs text-gray-400">{sub}</p>}
        </div>
    );
}

// Briefing panel
function BriefingPanel({ patientId }) {
    const [briefing, setBriefing] = useState(null);
    const [loading, setLoading] = useState(false);

    const fetch = useCallback(async () => {
        if (!patientId) return;
        setLoading(true);
        try {
            const data = await getBriefing(patientId);
            setBriefing(data);
        } finally {
            setLoading(false);
        }
    }, [patientId]);

    useEffect(() => { fetch(); }, [fetch]);

    return (
        <div className="card">
            <div className="flex items-center gap-2 mb-4">
                <Brain className="w-4 h-4 text-purple-500" />
                <h3 className="font-semibold text-gray-900 dark:text-white">AI Doctor Briefing</h3>
                {briefing?.briefingMode && (
                    <span className="ml-auto text-xs bg-purple-50 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 px-2 py-0.5 rounded-full font-medium border border-purple-100 dark:border-purple-800">
                        {briefing.briefingMode}
                    </span>
                )}
                <button
                    onClick={fetch}
                    disabled={loading}
                    className={clsx('btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5', !briefing?.briefingMode && 'ml-auto')}
                >
                    <RefreshCw className={clsx('w-3.5 h-3.5', loading && 'animate-spin')} />
                    Refresh
                </button>
            </div>

            {loading && !briefing ? (
                <div className="space-y-2 animate-pulse">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="h-4 bg-gray-100 dark:bg-gray-800 rounded" style={{ width: `${90 - i * 10}%` }} />
                    ))}
                </div>
            ) : briefing ? (
                <div>
                    <div className="flex items-center gap-2 mb-3">
                        <RiskBadge level={briefing.urgencyLevel ?? 'LOW'} />
                        {briefing.generatedAt && (
                            <span className="text-xs text-gray-400">
                                Generated {format(new Date(briefing.generatedAt), 'MMM d, HH:mm')}
                            </span>
                        )}
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-line">
                        {briefing.briefingText ?? briefing.summary}
                    </p>
                    {briefing.riskHighlights?.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
                            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
                                Risk Highlights
                            </p>
                            <ul className="space-y-1">
                                {briefing.riskHighlights.map((h, i) => (
                                    <li key={i} className="text-xs text-gray-600 dark:text-gray-300 flex items-start gap-1.5">
                                        <span className="text-red-500 mt-0.5">●</span>{h}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {briefing.trendFindings?.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
                            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
                                Trend Findings
                            </p>
                            <ul className="space-y-1">
                                {briefing.trendFindings.map((t, i) => (
                                    <li key={i} className="text-xs text-gray-600 dark:text-gray-300 flex items-start gap-1.5">
                                        <span className="text-amber-500 mt-0.5">▲</span>{t}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            ) : (
                <p className="text-sm text-gray-400">No briefing available. Click Refresh to generate.</p>
            )}
        </div>
    );
}

// ── Main page ─────────────────────────────────────────────────
export default function DoctorDashboard() {
    const { user } = useAuth();
    const {
        activePatientId, setActivePatientId,
        vitalsHistory, alerts, currentInference,
        loadingAlerts, loadingInference, loadPatientData,
    } = usePatient();

    const [searchInput, setSearchInput] = useState(activePatientId);
    const debouncedSearch = useDebounce(searchInput, 500);

    useEffect(() => {
        if (debouncedSearch && debouncedSearch !== activePatientId) {
            setActivePatientId(debouncedSearch);
        }
    }, [debouncedSearch, activePatientId, setActivePatientId]);

    const handleRefresh = () => {
        loadPatientData(activePatientId);
        toast.success('Refreshing patient data…');
    };

    // Latest vitals
    const latest = vitalsHistory.at(-1);

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
            <Navbar />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
                {/* Page header */}
                <div className="flex items-center justify-between flex-wrap gap-3">
                    <div>
                        <h1 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                            <Activity className="w-5 h-5 text-indigo-500" />
                            Doctor Dashboard
                        </h1>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                            Welcome back, {user?.name}
                        </p>
                    </div>
                    <button onClick={handleRefresh} className="btn-secondary flex items-center gap-2 text-sm">
                        <RefreshCw className="w-4 h-4" />
                        Refresh All
                    </button>
                </div>

                {/* Patient search */}
                <div className="card">
                    <div className="flex items-center gap-3 flex-wrap">
                        <div className="relative flex-1 min-w-52">
                            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                            <input
                                type="text"
                                value={searchInput}
                                onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
                                placeholder="Search patient ID — e.g. PAT-15554A87"
                                className="input pl-9 font-mono text-sm"
                            />
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                            <User className="w-4 h-4 text-gray-400" />
                            <span className="font-mono font-semibold text-gray-700 dark:text-gray-200">
                                {activePatientId}
                            </span>
                            {loadingInference
                                ? <Loader2 className="w-3.5 h-3.5 text-indigo-500 animate-spin" />
                                : <RiskBadge level={currentInference?.riskLevel ?? 'LOW'} size="sm" />}
                        </div>
                    </div>
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <StatCard
                        label="Heart Rate" value={latest?.heartRate?.toFixed(0)} unit="bpm"
                        sub="Last reading" color="indigo" icon={Activity}
                    />
                    <StatCard
                        label="SpO₂" value={latest?.spo2?.toFixed(1)} unit="%"
                        sub="Blood oxygen" color="emerald" icon={Zap}
                    />
                    <StatCard
                        label="Systolic BP" value={latest?.systolicBP?.toFixed(0)} unit="mmHg"
                        sub="Blood pressure" color="amber" icon={Activity}
                    />
                    <StatCard
                        label="Risk Score"
                        value={currentInference ? (currentInference.combinedRiskScore * 100).toFixed(0) : '—'}
                        unit="%"
                        sub={currentInference?.anomalyDetected ? '⚠ Anomaly' : 'Normal'}
                        color={currentInference?.riskLevel === 'HIGH' ? 'red' : 'indigo'}
                        icon={FileText}
                    />
                </div>

                {/* Charts */}
                <section>
                    <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-indigo-400" />
                        24-Hour Vitals Trends
                    </h2>
                    <VitalsChart data={vitalsHistory} />
                </section>

                {/* 2-col: Briefing + Alerts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="space-y-6">
                        <BriefingPanel patientId={activePatientId} />
                    </div>
                    <div>
                        <AlertsTable alerts={alerts} loading={loadingAlerts} />
                    </div>
                </div>


            </main>
        </div>
    );
}
