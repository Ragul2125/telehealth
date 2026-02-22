import { useState, useMemo } from 'react';
import { format } from 'date-fns';
import RiskBadge from './RiskBadge';
import { AlertTriangle, ChevronUp, ChevronDown, Search } from 'lucide-react';
import clsx from 'clsx';

const PAGE_SIZE = 8;

export default function AlertsTable({ alerts = [], loading = false }) {
    const [sortKey, setSortKey] = useState('timestamp');
    const [sortDir, setSortDir] = useState('desc');
    const [page, setPage] = useState(1);
    const [search, setSearch] = useState('');

    const toggleSort = (key) => {
        if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
        else { setSortKey(key); setSortDir('desc'); }
        setPage(1);
    };

    const filtered = useMemo(() => {
        const q = search.toLowerCase();
        return alerts.filter(
            (a) =>
                !q ||
                a.riskLevel?.toLowerCase().includes(q) ||
                a.reasons?.join(' ').toLowerCase().includes(q)
        );
    }, [alerts, search]);

    const sorted = useMemo(() => {
        return [...filtered].sort((a, b) => {
            let av = a[sortKey], bv = b[sortKey];
            if (sortKey === 'timestamp') { av = new Date(av); bv = new Date(bv); }
            if (sortKey === 'combinedRiskScore') { av = av ?? 0; bv = bv ?? 0; }
            if (av < bv) return sortDir === 'asc' ? -1 : 1;
            if (av > bv) return sortDir === 'asc' ? 1 : -1;
            return 0;
        });
    }, [filtered, sortKey, sortDir]);

    const paginated = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
    const totalPages = Math.ceil(sorted.length / PAGE_SIZE);

    const SortIcon = ({ k }) => {
        if (sortKey !== k) return <ChevronUp className="w-3.5 h-3.5 text-gray-300" />;
        return sortDir === 'asc'
            ? <ChevronUp className="w-3.5 h-3.5 text-indigo-500" />
            : <ChevronDown className="w-3.5 h-3.5 text-indigo-500" />;
    };

    if (loading) {
        return (
            <div className="card space-y-3">
                <div className="animate-pulse space-y-2">
                    {Array.from({ length: 5 }).map((_, i) => (
                        <div key={i} className="h-12 bg-gray-100 dark:bg-gray-800 rounded-xl" />
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
                <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                    <h3 className="font-semibold text-gray-900 dark:text-white">Alert History</h3>
                    <span className="bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-xs px-2 py-0.5 rounded-full font-medium">
                        {alerts.length}
                    </span>
                </div>
                <div className="relative">
                    <Search className="w-3.5 h-3.5 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                        type="text"
                        placeholder="Filter alerts…"
                        value={search}
                        onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                        className="input pl-8 py-1.5 text-sm w-48"
                    />
                </div>
            </div>

            {paginated.length === 0 ? (
                <div className="text-center py-10 text-gray-400 text-sm">
                    {search ? 'No alerts match your filter.' : 'No alerts recorded.'}
                </div>
            ) : (
                <div className="overflow-x-auto -mx-6">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                                {[
                                    { key: 'timestamp', label: 'Time' },
                                    { key: 'riskLevel', label: 'Risk' },
                                    { key: 'combinedRiskScore', label: 'Score' },
                                    { key: 'reasons', label: 'Triggers' },
                                ].map(({ key, label }) => (
                                    <th
                                        key={key}
                                        onClick={() => key !== 'reasons' && toggleSort(key)}
                                        className={clsx(
                                            'px-6 py-2.5 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide first:pl-6',
                                            key !== 'reasons' && 'cursor-pointer hover:text-indigo-500 transition-colors'
                                        )}
                                    >
                                        <span className="flex items-center gap-1">
                                            {label}
                                            {key !== 'reasons' && <SortIcon k={key} />}
                                        </span>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-50 dark:divide-gray-800/60">
                            {paginated.map((alert, idx) => (
                                <tr
                                    key={idx}
                                    className="hover:bg-gray-50/80 dark:hover:bg-gray-800/50 transition-colors"
                                >
                                    <td className="px-6 py-3 text-gray-500 dark:text-gray-400 whitespace-nowrap font-mono text-xs">
                                        {alert.timestamp
                                            ? format(new Date(alert.timestamp), 'MMM d, HH:mm:ss')
                                            : '—'}
                                    </td>
                                    <td className="px-6 py-3">
                                        <RiskBadge level={alert.riskLevel ?? 'LOW'} size="sm" />
                                    </td>
                                    <td className="px-6 py-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-1.5 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full rounded-full transition-all"
                                                    style={{
                                                        width: `${Math.min((alert.combinedRiskScore ?? 0) * 100, 100)}%`,
                                                        backgroundColor:
                                                            (alert.combinedRiskScore ?? 0) > 0.6 ? '#ef4444'
                                                                : (alert.combinedRiskScore ?? 0) > 0.35 ? '#f59e0b'
                                                                    : '#10b981',
                                                    }}
                                                />
                                            </div>
                                            <span className="text-xs text-gray-500">
                                                {((alert.combinedRiskScore ?? 0) * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-3 max-w-xs">
                                        {alert.reasons?.length ? (
                                            <ul className="space-y-0.5">
                                                {alert.reasons.slice(0, 2).map((r, i) => (
                                                    <li key={i} className="text-gray-600 dark:text-gray-300 text-xs truncate">
                                                        • {r}
                                                    </li>
                                                ))}
                                                {alert.reasons.length > 2 && (
                                                    <li className="text-gray-400 text-xs">+{alert.reasons.length - 2} more</li>
                                                )}
                                            </ul>
                                        ) : (
                                            <span className="text-gray-400 text-xs">—</span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-100 dark:border-gray-800">
                    <p className="text-xs text-gray-400">
                        Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, sorted.length)} of {sorted.length}
                    </p>
                    <div className="flex gap-1">
                        {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
                            <button
                                key={p}
                                onClick={() => setPage(p)}
                                className={clsx(
                                    'w-7 h-7 rounded-lg text-xs font-medium transition-colors',
                                    p === page
                                        ? 'bg-indigo-600 text-white'
                                        : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800'
                                )}
                            >
                                {p}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
