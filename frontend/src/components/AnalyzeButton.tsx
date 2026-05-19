interface AnalyzeButtonProps {
    drugCount: number;
    loading: boolean;
    onAnalyze: () => void;
}

export default function AnalyzeButton({ drugCount, loading, onAnalyze }: AnalyzeButtonProps) {
    const disabled = drugCount < 2 || loading

    return (
        <div className="analyze-section">
            <button
                id="analyze-button"
                className="analyze-btn"
                disabled={disabled}
                onClick={onAnalyze}
            >
                {loading ? (
                    <>
                        <span className="btn-spinner" />
                        Analyzing…
                    </>
                ) : (
                    <>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="btn-icon">
                            <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4" />
                        </svg>
                        Analyze Interactions
                    </>
                )}
            </button>
            {drugCount < 2 && (
                <p className="hint-text">Add at least 2 drugs to analyze interactions</p>
            )}
        </div>
    )
}
