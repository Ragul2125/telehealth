import { createContext, useContext, useState, useCallback } from 'react';

const AuthContext = createContext(null);

const MOCK_USERS = {
    doctor: {
        id: 'doc-001',
        name: 'Dr. Sarah Mitchell',
        role: 'doctor',
        token: 'mock-doctor-token-abc123',
        avatar: 'SM',
    },
    patient: {
        id: 'PAT-15554A87',
        name: 'James Wilson',
        role: 'patient',
        token: 'mock-patient-token-xyz789',
        avatar: 'JW',
    },
};

export function AuthProvider({ children }) {
    const [user, setUser] = useState(() => {
        const stored = localStorage.getItem('telehealth_user');
        return stored ? JSON.parse(stored) : null;
    });

    const login = useCallback((role, password) => {
        // Mock auth — in production this would hit an auth endpoint
        const mockUser = MOCK_USERS[role];
        if (!mockUser) return false;
        if (password !== 'demo123') return false;

        localStorage.setItem('telehealth_user', JSON.stringify(mockUser));
        localStorage.setItem('auth_token', mockUser.token);
        setUser(mockUser);
        return true;
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem('telehealth_user');
        localStorage.removeItem('auth_token');
        setUser(null);
    }, []);

    return (
        <AuthContext.Provider value={{ user, login, logout, isAuthenticated: !!user }}>
            {children}
        </AuthContext.Provider>
    );
}

export const useAuth = () => {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider');
    return ctx;
};
