export default function LoadingOverlay() {
    return (
        <div className="loading-overlay">
            <div className="loading-content">
                <div className="pulse-ring" />
                <div className="pulse-dot" />
                <p className="loading-text">Analyzing drug interactions…</p>
                <p className="loading-sub">Querying knowledge base & classifying severity</p>
            </div>
        </div>
    )
}
