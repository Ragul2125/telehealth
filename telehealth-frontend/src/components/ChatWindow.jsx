import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Stethoscope } from 'lucide-react';
import { sendTriageMessage } from '../api/triageApi';
import clsx from 'clsx';

const SUGGESTED = [
    'I feel chest pain',
    'I feel dizzy',
    'I have a headache',
    'What do my latest readings mean?',
];

export default function ChatWindow({ patientId }) {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            text: "Hello! I'm your clinical AI assistant. I can help you understand your vitals and answer health-related questions based on your monitoring data. How can I assist you today?",
            ts: new Date(),
        },
    ]);
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const endRef = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const send = async (text) => {
        const trimmed = (text ?? input).trim();
        if (!trimmed || sending) return;

        setInput('');
        setSending(true);

        const userMsg = { role: 'user', text: trimmed, ts: new Date() };
        setMessages((prev) => [...prev, userMsg]);

        try {
            const response = await sendTriageMessage(patientId, trimmed);
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', text: response, ts: new Date() },
            ]);
        } finally {
            setSending(false);
            setTimeout(() => inputRef.current?.focus(), 100);
        }
    };

    const handleKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    };

    const fmt = (d) =>
        d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return (
        <div className="card flex flex-col h-[520px] p-0 overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-100 dark:border-gray-800 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/40 dark:to-purple-950/40">
                <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-md shadow-indigo-500/30">
                    <Stethoscope className="w-4.5 h-4.5 text-white" />
                </div>
                <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white text-sm">Triage AI Assistant</h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Clinical decision support · Not a replacement for care</p>
                </div>
                <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                    Online
                </span>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 scroll-smooth">
                {messages.map((msg, i) => (
                    <div
                        key={i}
                        className={clsx(
                            'flex gap-2.5 max-w-[88%]',
                            msg.role === 'user' && 'ml-auto flex-row-reverse'
                        )}
                    >
                        {/* Avatar */}
                        <div
                            className={clsx(
                                'w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5',
                                msg.role === 'assistant'
                                    ? 'bg-indigo-100 dark:bg-indigo-900/40'
                                    : 'bg-gray-200 dark:bg-gray-700'
                            )}
                        >
                            {msg.role === 'assistant'
                                ? <Bot className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                                : <User className="w-3.5 h-3.5 text-gray-600 dark:text-gray-300" />}
                        </div>

                        {/* Bubble */}
                        <div>
                            <div
                                className={clsx(
                                    'px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed',
                                    msg.role === 'assistant'
                                        ? 'bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-sm shadow-sm'
                                        : 'bg-indigo-600 text-white rounded-tr-sm'
                                )}
                            >
                                {msg.text}
                            </div>
                            <p className={clsx(
                                'text-xs text-gray-400 mt-1',
                                msg.role === 'user' && 'text-right'
                            )}>
                                {fmt(msg.ts)}
                            </p>
                        </div>
                    </div>
                ))}

                {/* Typing indicator */}
                {sending && (
                    <div className="flex gap-2.5">
                        <div className="w-7 h-7 rounded-full bg-indigo-100 dark:bg-indigo-900/40 flex items-center justify-center">
                            <Bot className="w-3.5 h-3.5 text-indigo-600" />
                        </div>
                        <div className="bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm flex items-center gap-1.5">
                            {[0, 1, 2].map((i) => (
                                <span
                                    key={i}
                                    className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
                                    style={{ animationDelay: `${i * 150}ms` }}
                                />
                            ))}
                        </div>
                    </div>
                )}
                <div ref={endRef} />
            </div>

            {/* Suggestions */}
            {messages.length <= 2 && (
                <div className="px-4 pb-2 flex gap-1.5 flex-wrap">
                    {SUGGESTED.map((s) => (
                        <button
                            key={s}
                            onClick={() => send(s)}
                            className="text-xs px-3 py-1.5 rounded-full bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-900/50 transition-colors border border-indigo-100 dark:border-indigo-800"
                        >
                            {s}
                        </button>
                    ))}
                </div>
            )}

            {/* Input */}
            <div className="px-4 pb-4 pt-2 border-t border-gray-100 dark:border-gray-800">
                <div className="flex gap-2 items-end">
                    <textarea
                        ref={inputRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKey}
                        placeholder="Ask about your vitals or symptoms…"
                        rows={1}
                        className="input resize-none flex-1 text-sm py-2.5 max-h-28 overflow-y-auto"
                        style={{ fieldSizing: 'content' }}
                    />
                    <button
                        onClick={() => send()}
                        disabled={!input.trim() || sending}
                        className="btn-primary w-10 h-10 p-0 flex items-center justify-center shrink-0 rounded-xl"
                        aria-label="Send"
                    >
                        {sending
                            ? <Loader2 className="w-4 h-4 animate-spin" />
                            : <Send className="w-4 h-4" />}
                    </button>
                </div>
                <p className="text-xs text-gray-400 mt-1.5 text-center">
                    ⚕️ Clinical AI support only — not a substitute for professional medical advice
                </p>
            </div>
        </div>
    );
}
