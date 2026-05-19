import { useState } from 'react'
import DrugSearch from './components/DrugSearch'
import SelectedDrugs from './components/SelectedDrugs'
import AnalyzeButton from './components/AnalyzeButton'
import ResultsPanel from './components/ResultsPanel'
import LoadingOverlay from './components/LoadingOverlay'
import { analyzeInteractions } from './api'
import type { AnalyzeResponse } from './api'

export default function App() {
    const [drugs, setDrugs] = useState<string[]>([])
    const [result, setResult] = useState<AnalyzeResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    function addDrug(name: string) {
        const normalized = name.toLowerCase().trim()
        if (normalized && !drugs.includes(normalized)) {
            setDrugs((prev) => [...prev, normalized])
        }
    }

    function removeDrug(name: string) {
        setDrugs((prev) => prev.filter((d) => d !== name))
    }

    async function handleAnalyze() {
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const data = await analyzeInteractions(drugs)
            setResult(data)
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Something went wrong')
        } finally {
            setLoading(false)
        }
    }

    return (
        <>
            {loading && <LoadingOverlay />}

            <div className="app-container">
                {/* Header */}
                <header className="app-header">
                    <div className="header-glow" />
                    <div className="header-content">
                        <div className="logo-row">
                            <div className="logo-icon">
                                <svg viewBox="0 0 32 32" fill="none">
                                    <rect x="4" y="10" width="24" height="12" rx="6" stroke="url(#lg)" strokeWidth="2.5" />
                                    <line x1="16" y1="10" x2="16" y2="22" stroke="url(#lg)" strokeWidth="2" strokeDasharray="2 2" />
                                    <defs>
                                        <linearGradient id="lg" x1="0" y1="0" x2="32" y2="32">
                                            <stop stopColor="#818cf8" />
                                            <stop offset="1" stopColor="#c084fc" />
                                        </linearGradient>
                                    </defs>
                                </svg>
                            </div>
                            <h1>Drug Interaction Analysis</h1>
                        </div>
                        <p className="header-sub">
                            AI-powered clinical decision support — search drugs, analyze interactions, and review evidence-based severity classifications.
                        </p>
                    </div>
                </header>

                {/* Main content */}
                <main className="main-content">
                    <section className="input-section glass-card">
                        <DrugSearch onAddDrug={addDrug} selectedDrugs={drugs} />
                        <SelectedDrugs drugs={drugs} onRemove={removeDrug} />
                        <AnalyzeButton
                            drugCount={drugs.length}
                            loading={loading}
                            onAnalyze={handleAnalyze}
                        />
                    </section>

                    {error && (
                        <div className="error-banner">
                            <span className="error-icon">⚠️</span>
                            <span>{error}</span>
                            <button className="error-dismiss" onClick={() => setError(null)}>×</button>
                        </div>
                    )}

                    <ResultsPanel result={result} />
                </main>

                {/* Footer */}
                <footer className="app-footer">
                    <p>Drug Interaction AI · For informational purposes only · Not a substitute for professional medical advice</p>
                </footer>
            </div>
        </>
    )
}
