import { useState, useRef, useEffect, useCallback } from 'react'
import { searchDrugs } from '../api'

interface DrugSearchProps {
    onAddDrug: (drug: string) => void;
    selectedDrugs: string[];
}

export default function DrugSearch({ onAddDrug, selectedDrugs }: DrugSearchProps) {
    const [query, setQuery] = useState('')
    const [suggestions, setSuggestions] = useState<string[]>([])
    const [showDropdown, setShowDropdown] = useState(false)
    const [highlightIdx, setHighlightIdx] = useState(-1)
    const [loading, setLoading] = useState(false)
    const inputRef = useRef<HTMLInputElement>(null)
    const dropdownRef = useRef<HTMLDivElement>(null)
    const debounceRef = useRef<ReturnType<typeof setTimeout>>()

    const fetchSuggestions = useCallback(async (q: string) => {
        if (q.length < 2) {
            setSuggestions([])
            setShowDropdown(false)
            return
        }
        setLoading(true)
        try {
            const results = await searchDrugs(q)
            const filtered = results.filter(
                (s) => !selectedDrugs.includes(s.toLowerCase())
            )
            setSuggestions(filtered)
            setShowDropdown(filtered.length > 0)
            setHighlightIdx(-1)
        } catch {
            setSuggestions([])
        } finally {
            setLoading(false)
        }
    }, [selectedDrugs])

    useEffect(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current)
        debounceRef.current = setTimeout(() => fetchSuggestions(query), 300)
        return () => {
            if (debounceRef.current) clearTimeout(debounceRef.current)
        }
    }, [query, fetchSuggestions])

    // Close dropdown on outside click
    useEffect(() => {
        function handleClick(e: MouseEvent) {
            if (
                dropdownRef.current &&
                !dropdownRef.current.contains(e.target as Node) &&
                inputRef.current &&
                !inputRef.current.contains(e.target as Node)
            ) {
                setShowDropdown(false)
            }
        }
        document.addEventListener('mousedown', handleClick)
        return () => document.removeEventListener('mousedown', handleClick)
    }, [])

    function selectDrug(name: string) {
        onAddDrug(name.toLowerCase())
        setQuery('')
        setSuggestions([])
        setShowDropdown(false)
        inputRef.current?.focus()
    }

    function handleKeyDown(e: React.KeyboardEvent) {
        if (!showDropdown) return
        if (e.key === 'ArrowDown') {
            e.preventDefault()
            setHighlightIdx((i) => Math.min(i + 1, suggestions.length - 1))
        } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setHighlightIdx((i) => Math.max(i - 1, 0))
        } else if (e.key === 'Enter' && highlightIdx >= 0) {
            e.preventDefault()
            selectDrug(suggestions[highlightIdx])
        } else if (e.key === 'Escape') {
            setShowDropdown(false)
        }
    }

    return (
        <div className="drug-search">
            <div className="search-input-wrapper">
                <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <input
                    ref={inputRef}
                    id="drug-search-input"
                    type="text"
                    placeholder="Search for a drug (e.g. warfarin, ibuprofen)…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
                    autoComplete="off"
                />
                {loading && <span className="search-spinner" />}
            </div>

            {showDropdown && (
                <div className="suggestions-dropdown" ref={dropdownRef}>
                    {suggestions.map((s, i) => (
                        <button
                            key={s}
                            className={`suggestion-item${i === highlightIdx ? ' highlighted' : ''}`}
                            onMouseDown={() => selectDrug(s)}
                            onMouseEnter={() => setHighlightIdx(i)}
                        >
                            <svg className="pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="3" y="11" width="18" height="6" rx="3" />
                                <rect x="3" y="7" width="18" height="10" rx="5" />
                            </svg>
                            {s}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}
