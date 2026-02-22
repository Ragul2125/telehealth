# TeleHealth AI — Frontend

> **Cloud-based Telehealth ML + GenAI patient monitoring platform**  
> React · Vite · TailwindCSS · Recharts · Axios

---

## Features

- 🩺 **Doctor Dashboard** — Patient search, 24h vitals charts, AI briefings, alert history
- 👤 **Patient Dashboard** — Live vitals, risk status, submission form, AI triage chat
- 🧠 **AI Triage Chat** — Clinical decision support (POST `/triage`)
- 📊 **Recharts Graphs** — Heart rate, SpO₂, Systolic BP, Risk score (dynamic)
- 🔐 **Role-Based Auth** — Doctor / Patient mock login with protected routes
- 🌙 **Dark Mode** — One-click toggle, persisted in localStorage
- ⚡ **Offline Fallback** — Demo data shown when API is unreachable
- 🔄 **Retry + Toasts** — Auto-retry on 5xx, user-friendly error toasts

---

## Project Structure

```
telehealth-frontend/
├── src/
│   ├── api/
│   │   ├── axiosClient.js      # Axios + interceptors + retry
│   │   ├── vitalsApi.js        # POST /vitals, POST /inference
│   │   ├── alertsApi.js        # GET  /alerts/:patientId
│   │   ├── briefingApi.js      # GET  /brief/:patientId
│   │   └── triageApi.js        # POST /triage
│   ├── context/
│   │   ├── AuthContext.jsx     # Role auth (doctor/patient)
│   │   └── PatientContext.jsx  # Active patient data, vitals state
│   ├── components/
│   │   ├── Navbar.jsx          # Top nav, dark toggle, logout
│   │   ├── RiskBadge.jsx       # Color-coded risk indicator
│   │   ├── VitalsChart.jsx     # 4x Recharts line charts
│   │   ├── AlertsTable.jsx     # Sortable, paginated alert table
│   │   ├── ChatWindow.jsx      # Triage AI chat interface
│   │   └── VitalsForm.jsx      # Vitals submission + inference
│   ├── pages/
│   │   ├── Login.jsx           # Role selector + mock login
│   │   ├── DoctorDashboard.jsx # Full monitoring dashboard
│   │   └── PatientDashboard.jsx# Patient self-monitoring view
│   ├── App.jsx                 # BrowserRouter + routes + Toaster
│   └── main.jsx
├── .env
├── index.html
└── package.json
```

---

## Installation & Setup

### Prerequisites
- Node.js ≥ 18
- npm ≥ 9

### 1. Clone / enter directory
```bash
cd telehealth-frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Configure environment
```bash
# .env
VITE_API_BASE_URL=https://api.example.com   # your AWS API Gateway URL
```

### 4. Run locally
```bash
npm run dev
# → http://localhost:5173
```

### Demo login
| Role    | Password  |
|---------|-----------|
| Doctor  | `demo123` |
| Patient | `demo123` |

---

## Build for Production

```bash
npm run build
# Output: dist/
```

Preview the production build locally:
```bash
npm run preview
```

---

## Deploy to AWS S3 + CloudFront

### 1. Create S3 bucket (static website hosting)
```bash
aws s3 mb s3://telehealth-ai-frontend --region ap-south-1
aws s3 website s3://telehealth-ai-frontend \
  --index-document index.html \
  --error-document index.html      # SPA fallback
```

### 2. Set bucket policy (public read)
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::telehealth-ai-frontend/*"
  }]
}
```

### 3. Upload dist/
```bash
aws s3 sync dist/ s3://telehealth-ai-frontend --delete
```

### 4. CloudFront distribution
- **Origin**: `telehealth-ai-frontend.s3-website.ap-south-1.amazonaws.com`
- **Default root object**: `index.html`
- **Error page**: `/index.html` (HTTP 403 → 200) — required for React Router
- **Cache policy**: CachingDisabled for API origins, CachingOptimized for static assets
- **HTTPS**: Use `us-east-1` ACM certificate for custom domain

### 5. CORS configuration (API Gateway)
Add origin `https://your-cloudfront-domain.cloudfront.net` to your API Gateway CORS allowed origins.

### 6. Invalidate CloudFront cache after deploy
```bash
aws cloudfront create-invalidation \
  --distribution-id E1EXAMPLE123 \
  --paths "/*"
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | AWS API Gateway base URL | `https://api.example.com` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/vitals` | Submit patient vitals |
| POST | `/inference` | Run ML risk inference |
| GET | `/alerts/:patientId` | Get patient alert history |
| GET | `/brief/:patientId` | Get AI doctor briefing |
| POST | `/triage` | Triage AI chat |

---

## Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| UI Framework | React | 19 |
| Build Tool | Vite | 7 |
| Styling | TailwindCSS | 4 |
| HTTP Client | Axios | latest |
| Routing | React Router | 7 |
| Charts | Recharts | latest |
| Icons | Lucide React | latest |
| Notifications | React Hot Toast | latest |
| Dates | date-fns | latest |

---

## License

MIT — For demonstration and educational purposes only.  
⚕️ Not intended for clinical use.
