import { useState } from 'react'
import DrugSearch from './components/DrugSearch'
import SelectedDrugs from './components/SelectedDrugs'
import AnalyzeButton from './components/AnalyzeButton'
import ResultsPanel from './components/ResultsPanel'
import LoadingOverlay from './components/LoadingOverlay'
import { analyzeInteractions, DEMO_MODE } from './api'
import type { AnalyzeResponse } from './api'
import MedicationReview from './MedicationReview'

function ResearchApp() {
    const [drugs, setDrugs] = useState<string[]>([])
    const [result, setResult] = useState<AnalyzeResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    function addDrug(name: string) {
        const normalized = name.toLowerCase().trim()
        if (normalized && !drugs.includes(normalized)) {
            setResult(null)
            setError(null)
            setDrugs((prev) => [...prev, normalized])
        }
    }

    function removeDrug(name: string) {
        setResult(null)
        setError(null)
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
                            {DEMO_MODE ? 'Browser evidence demo · Local fixture lookup · No account required' : 'Research prototype for interaction retrieval, classification, and explanation.'}
                        </p>
                    </div>
                </header>

                {/* Main content */}
                <main className="main-content">
                    <section className="input-section glass-card">
                        {DEMO_MODE && <div className="demo-notice">
                            <strong>Research fixtures, not clinical advice.</strong>
                            <p>This public demo runs locally in your browser. It does not run the Python API, trained classifier, or LLM. Labels and text are unvalidated examples from the repository.</p>
                            <button type="button" onClick={() => { setDrugs(['warfarin', 'ibuprofen']); setResult(null); setError(null); }}>Load an example pair</button>
                            <a href="https://github.com/vajja1405/drug-interaction-rag-chatbot" target="_blank" rel="noopener noreferrer">Source & full Python system ↗</a>
                            <a href="https://astra6-drug-interaction-ai.hf.space" target="_blank" rel="noopener noreferrer">Open the medication label review ↗</a>
                        </div>}
                        {!DEMO_MODE && <div className="demo-notice">
                            <strong>Full research application · Python API + retrieval + model explanation</strong>
                            <p>Enter medication names only. Names are sent to this server and its model provider. The evidence corpus is curated and unvalidated; do not use the output for care decisions.</p>
                            <button type="button" onClick={() => { setDrugs(['warfarin', 'ibuprofen']); setResult(null); setError(null); }}>Load an example pair</button>
                        </div>}
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

export default function App() {
    const [mode,setMode]=useState('review')
    if(DEMO_MODE)return <ResearchApp/>
    return <><nav className="mode-nav" aria-label="Application mode"><button aria-pressed={mode==='review'} onClick={()=>setMode('review')}>Medication label review</button><button aria-pressed={mode==='research'} onClick={()=>setMode('research')}>RAG research demo</button><a href="https://github.com/vajja1405/drug-interaction-rag-chatbot" target="_blank" rel="noreferrer">View source ↗</a></nav>{mode==='review'?<MedicationReview/>:<ResearchApp/>}</>
}
