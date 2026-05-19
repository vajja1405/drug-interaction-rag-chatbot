interface SelectedDrugsProps {
    drugs: string[];
    onRemove: (drug: string) => void;
}

export default function SelectedDrugs({ drugs, onRemove }: SelectedDrugsProps) {
    if (drugs.length === 0) return null

    return (
        <div className="selected-drugs">
            <span className="selected-label">Selected drugs</span>
            <div className="chips-container">
                {drugs.map((drug) => (
                    <span key={drug} className="drug-chip">
                        <svg className="chip-pill" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="7" width="18" height="10" rx="5" />
                        </svg>
                        {drug}
                        <button
                            className="chip-remove"
                            onClick={() => onRemove(drug)}
                            aria-label={`Remove ${drug}`}
                            title={`Remove ${drug}`}
                        >
                            ×
                        </button>
                    </span>
                ))}
            </div>
        </div>
    )
}
