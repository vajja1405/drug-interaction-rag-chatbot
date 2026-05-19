import { useState } from 'react'
import type { AnalyzeResponse, DrugInteraction } from '../api'

interface ResultsPanelProps {
    result: AnalyzeResponse | null;
}

const SEVERITY_CONFIG: Record<string, { color: string; bg: string; icon: string }> = {
    Severe: { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)', icon: '🔴' },
    Moderate: { color: '#ffaa2c', bg: 'rgba(255,170,44,0.12)', icon: '🟠' },
    Low: { color: '#4da6ff', bg: 'rgba(77,166,255,0.12)', icon: '🔵' },
    Unknown: { color: '#888', bg: 'rgba(136,136,136,0.12)', icon: '⚪' },
}

function SeveritySummary({ interactions }: { interactions: DrugInteraction[] }) {
    const counts: Record<string, number> = {}
    for (const ix of interactions) {
        counts[ix.severity] = (counts[ix.severity] || 0) + 1
    }

    return (
        <div className="severity-summary">
            {Object.entries(SEVERITY_CONFIG).map(([level, cfg]) => {
                const c = counts[level] || 0
                if (c === 0 && level === 'Unknown') return null
                return (
                    <div key={level} className="severity-count" style={{ background: cfg.bg, borderColor: cfg.color }}>
                        <span className="severity-icon">{cfg.icon}</span>
                        <span className="severity-number">{c}</span>
                        <span className="severity-label">{level}</span>
                    </div>
                )
            })}
        </div>
    )
}

function InteractionCard({ interaction }: { interaction: DrugInteraction }) {
    const [open, setOpen] = useState(false)
    const cfg = SEVERITY_CONFIG[interaction.severity] ?? SEVERITY_CONFIG.Unknown

    return (
        <div className="interaction-card" style={{ borderLeftColor: cfg.color }}>
            <button className="card-header" onClick={() => setOpen(!open)}>
                <div className="card-title-row">
                    <span className="drug-pair">
                        {interaction.pair[0]} <span className="pair-sep">×</span> {interaction.pair[1]}
                    </span>
                    <span className="severity-badge" style={{ background: cfg.bg, color: cfg.color }}>
                        {cfg.icon} {interaction.severity}
                    </span>
                </div>
                <span className={`chevron ${open ? 'open' : ''}`}>▾</span>
            </button>

            {open && (
                <div className="card-body">
                    {interaction.mechanism && (
                        <div className="detail-section">
                            <h4>⚙️ Mechanism</h4>
                            <p>{interaction.mechanism}</p>
                        </div>
                    )}
                    {interaction.clinical_effects && (
                        <div className="detail-section">
                            <h4>🩺 Clinical Effects</h4>
                            <p>{interaction.clinical_effects}</p>
                        </div>
                    )}
                    {interaction.management && (
                        <div className="detail-section">
                            <h4>📋 Management</h4>
                            <p>{interaction.management}</p>
                        </div>
                    )}
                    {interaction.monitoring.length > 0 && (
                        <div className="detail-section">
                            <h4>📊 Monitoring</h4>
                            <ul>
                                {interaction.monitoring.map((m, i) => (
                                    <li key={i}>{m}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {interaction.source_url && (
                        <div className="detail-section citation-section">
                            <h4>📎 Source</h4>
                            <a
                                href={interaction.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="citation-link"
                            >
                                {interaction.source_page || interaction.source_name || 'View Source'}
                            </a>
                            {interaction.source_name && (
                                <span className="citation-badge">{interaction.source_name}</span>
                            )}
                        </div>
                    )}
                    <div className="detail-meta">
                        <span>Method: {interaction.classification_method}</span>
                        <span>Confidence: {(interaction.confidence * 100).toFixed(0)}%</span>
                        {interaction.interaction_type && (
                            <span>Type: {interaction.interaction_type}</span>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}

export default function ResultsPanel({ result }: ResultsPanelProps) {
    if (!result) return null

    const { interactions } = result

    return (
        <div className="results-panel">
            {/* Overall risk banner */}
            <div className="risk-banner" style={{
                background: (SEVERITY_CONFIG[result.overall_risk] ?? SEVERITY_CONFIG.Unknown).bg,
                borderColor: (SEVERITY_CONFIG[result.overall_risk] ?? SEVERITY_CONFIG.Unknown).color,
            }}>
                <div className="risk-banner-content">
                    <span className="risk-level">
                        Overall Risk: <strong>{result.overall_risk}</strong>
                    </span>
                    <span className="risk-meta">
                        {result.pairs_checked.length} pair{result.pairs_checked.length !== 1 ? 's' : ''} checked
                        · {result.retrieved_docs} docs retrieved
                        · {result.processing_time_ms.toFixed(0)}ms
                    </span>
                </div>
            </div>

            {/* Severity summary counters */}
            <SeveritySummary interactions={interactions} />

            {/* Interaction cards */}
            {interactions.length === 0 ? (
                <div className="no-interactions">
                    <div className="no-interactions-icon">✅</div>
                    <h3>No interactions found</h3>
                    <p>No clinically significant interactions were identified between the selected drugs.</p>
                </div>
            ) : (
                <div className="interactions-list">
                    {interactions.map((ix, i) => (
                        <InteractionCard key={`${ix.pair.join('-')}-${i}`} interaction={ix} />
                    ))}
                </div>
            )}

            {/* Narrative explanation */}
            {result.explanation && (
                <div className="narrative-section">
                    <h3>📖 Clinical Summary</h3>
                    <p>{result.explanation}</p>
                </div>
            )}

            {/* Monitoring priorities */}
            {result.monitoring_priorities.length > 0 && (
                <div className="monitoring-section">
                    <h3>🔍 Monitoring Priorities</h3>
                    <ul>
                        {result.monitoring_priorities.map((m, i) => (
                            <li key={i}>{m}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Disclaimer */}
            <p className="disclaimer">{result.disclaimer}</p>
        </div>
    )
}
