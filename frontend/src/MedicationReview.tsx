import { useEffect, useRef, useState } from 'react'
import { requestJson, statusNames } from './review-api'
import type { Medication, CatalogInfo, Review, Pair } from './review-api'
import './review.css'

const examples: Medication[] = [
    {rxcui:'11289',name:'warfarin',tty:'IN'}, {rxcui:'5640',name:'ibuprofen',tty:'IN'},
    {rxcui:'703',name:'amiodarone',tty:'IN'}, {rxcui:'1191',name:'aspirin',tty:'IN'},
]
const typeName: Record<string,string> = {IN:'Ingredient',MIN:'Combination',BN:'Brand'}

function PairCard({pair}: {pair: Pair}) {
    return <details className={`review-pair ${pair.status}`}>
        <summary><span>{pair.names.join(' + ')}</span><span className="review-badge">{statusNames[pair.status]}</span></summary>
        <div className="review-pair-body">
            {!pair.source_complete && <p className="review-muted">This pair has incomplete source coverage, including when a mention was found in another label.</p>}
            {!!pair.shared_ingredients.length && <p><strong>Possible shared ingredients:</strong> {pair.shared_ingredients.join(', ')}. Brand mappings can span formulations. Confirm the exact package and intended use with a pharmacist.</p>}
            {pair.evidence.length > 0 ? <>
                <p className="review-muted">These passages mention a selected name or ingredient. A mention may describe no effect or a conditional finding; no severity or clinical conclusion has been assigned.</p>
                {pair.evidence.map((e,i)=><figure key={i}><blockquote>{e.excerpt}</blockquote><figcaption>{e.section} · {e.source_drug} · label version {e.version}<br/><a href={e.source_url} target="_blank" rel="noreferrer">Read the full label: {e.source_title} ↗</a></figcaption></figure>)}
                {pair.evidence_truncated && <p>Additional matches are omitted. Open the original labels for the full context.</p>}
            </> : <p>{pair.status === 'incomplete_sources' ? 'At least one interaction section could not be retrieved. This pair has incomplete source coverage.' : 'No direct name match was found in the sampled interaction sections. Drug-class references and other sources may still describe interactions.'} <strong>This does not mean the combination is safe.</strong></p>}
        </div>
    </details>
}

export default function MedicationReview() {
    const [catalog,setCatalog]=useState<CatalogInfo|null>(null)
    const [query,setQuery]=useState('')
    const [suggestions,setSuggestions]=useState<Medication[]>([])
    const [selected,setSelected]=useState<Medication[]>([])
    const [searching,setSearching]=useState(false)
    const [searchError,setSearchError]=useState('')
    const [error,setError]=useState('')
    const [loading,setLoading]=useState(false)
    const [elapsed,setElapsed]=useState(0)
    const [result,setResult]=useState<Review|null>(null)
    const [filter,setFilter]=useState('all')
    const [reportText,setReportText]=useState('')
    const [page,setPage]=useState(0)
    const [matrix,setMatrix]=useState(false)
    const [focusPair,setFocusPair]=useState<Pair|null>(null)
    const controller=useRef<AbortController|null>(null)
    useEffect(()=>{const c=new AbortController(); requestJson<CatalogInfo>('/api/v2/catalog',{signal:c.signal}).then(setCatalog).catch(e=>{if(!c.signal.aborted)setError(e.message)});return()=>{c.abort();controller.current?.abort()}},[])
    useEffect(()=>{
        const c=new AbortController();setSearchError('');setSuggestions([])
        if(query.trim().length<2){setSearching(false);return()=>c.abort()}
        setSearching(true)
        const timer=setTimeout(()=>{requestJson<{results:Medication[]}>(`/api/v2/medications?q=${encodeURIComponent(query.trim())}`,{signal:c.signal}).then(d=>{if(!c.signal.aborted)setSuggestions(d.results)}).catch(e=>{if(!c.signal.aborted)setSearchError(e.message)}).finally(()=>{if(!c.signal.aborted)setSearching(false)})},250)
        return()=>{clearTimeout(timer);c.abort()}
    },[query])
    useEffect(()=>{if(!loading)return;const start=Date.now();setElapsed(0);const t=setInterval(()=>setElapsed(Math.floor((Date.now()-start)/1000)),1000);return()=>clearInterval(t)},[loading])
    function clearResult(){setReportText('');setResult(null);setError('');setFocusPair(null);setPage(0);setFilter('all')}
    function add(m:Medication){if(selected.length>=20||selected.some(x=>x.rxcui===m.rxcui))return;setSelected([...selected,m]);setQuery('');clearResult()}
    async function review(){
        const c=new AbortController();controller.current=c;setLoading(true);clearResult()
        const timer=setTimeout(()=>c.abort(),125000)
        try{const d=await requestJson<Review>('/api/v2/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rxcuis:selected.map(m=>m.rxcui)}),signal:c.signal});if(!c.signal.aborted)setResult(d)}
        catch(e){setError(c.signal.aborted?'Review stopped. No result or safety conclusion was produced.':e instanceof Error?e.message:'Review unavailable.')}
        finally{clearTimeout(timer);setLoading(false)}
    }
    function download(){
        if(!result)return
        const lines=['MEDICATION LABEL REVIEW','For a conversation with a pharmacist; not a treatment or safety assessment.',`Prepared: ${result.generated_at}`,result.method,'',...result.medications.map(m=>`${m.name} (RxCUI ${m.rxcui}): ${statusNames[m.status]}`),'',...result.pairs.flatMap(p=>[`${p.names.join(' + ')} — ${statusNames[p.status]}${p.source_complete?'':' (incomplete source coverage)'}`,...p.evidence.map(e=>`${e.excerpt}\n${e.source_title}\n${e.source_url}`)]),'','SOURCES',...result.medications.flatMap(m=>m.labels.map(l=>`${l.title}\nVersion ${l.version}; published ${l.published_date}\n${l.url}`)),'','LIMITATIONS',...result.limitations,'','QUESTIONS TO DISCUSS','Does this list match the exact strengths and formulations I take?','Do any class-based, condition-specific or multi-drug effects need review?','Are any shared ingredients intentional?']
        const text=lines.join('\n');setReportText(text)
        const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='medication-label-review.txt';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000)
    }
    const pairs=result?.pairs.filter(p=>filter==='all'||(filter==='incomplete_sources'?!p.source_complete:p.status===filter))||[]
    const totalPages=Math.ceil(pairs.length/15)
    return <div className="med-review">
        <header className="review-hero"><p className="review-eyebrow">DRUG INTERACTION AI / MEDICATION REVIEW</p><h1>Your medicines.<br/><span>The evidence, together.</span></h1><p>Build a medication list, inspect label passages and prepare better questions for your pharmacist.</p>
            <div className="review-capabilities"><span>{catalog?catalog.terms.toLocaleString():'…'} searchable terms</span><span>Up to 20 medicines</span><span>Up to 190 pair comparisons</span></div>
        </header>
        <main>
            <aside className="review-notice"><strong>A label review, not a safety clearance.</strong> No result on this page certifies that medicines are safe to combine. This tool does not assess dose, health conditions or three-drug and higher-order effects. Do not start, stop or change a medicine based on this page.</aside>
            <section className="review-builder" aria-labelledby="list-title"><div className="review-section-top"><div><p className="review-eyebrow">01 / BUILD YOUR LIST</p><h2 id="list-title">What would you like to review?</h2></div><button disabled={loading} className="review-secondary" onClick={()=>{setSelected(examples);setQuery('');clearResult()}}>Try a four-medicine example</button></div>
                <label htmlFor="medication-query">Search an ingredient, combination or brand</label>
                <input id="medication-query" type="search" autoComplete="off" maxLength={80} value={query} disabled={loading||selected.length>=20} placeholder="Try metformin, Eliquis or acetaminophen…" onChange={e=>setQuery(e.target.value)}/>
                <div aria-live="polite" className="review-search-status">{searching?'Searching the terminology catalog…':searchError|| (query.trim().length>=2 && !suggestions.length?'No matching term. Check the spelling; unresolved medicines cannot be reviewed.':'Select a catalog result to add it. A recognized name does not guarantee label or interaction coverage.')}</div>
                {!!suggestions.length&&<ul className="review-suggestions">{suggestions.map(m=><li key={m.rxcui}><button disabled={selected.some(x=>x.rxcui===m.rxcui)} onClick={()=>add(m)}><span>{m.name}</span><small>{typeName[m.tty]} · {m.rxcui}</small><span>＋</span></button></li>)}</ul>}
                <div className="review-selected">{selected.map(m=><span key={m.rxcui}>{m.name}<button disabled={loading} aria-label={`Remove ${m.name}`} onClick={()=>{setSelected(selected.filter(x=>x.rxcui!==m.rxcui));clearResult()}}>×</button></span>)}</div>
                <div className="review-actions"><span>{selected.length} / 20 selected · {selected.length*(selected.length-1)/2} possible pairs</span><button className="review-primary" disabled={selected.length<2||loading} onClick={review}>{loading?`Retrieving labels… ${elapsed}s`:'Review medication labels →'}</button>{loading&&<button className="review-secondary" onClick={()=>controller.current?.abort()}>Cancel</button>}</div>
                <p className="review-privacy">Selected medication identifiers go to this server and NLM services. This review uses no language-model provider and does not save your list in browser storage. Hosting and upstream services may keep operational logs. Do not enter names, medical records or other identifying information.</p>
            </section>
            {error&&<div role="alert" className="review-error">{error}</div>}
            {loading&&<p role="status" className="review-loading">Fetching a sample human label for each selected concept, then matching ingredient names. Larger lists can take up to two minutes. Missing sources will be identified explicitly.</p>}
            {result&&<section className="review-results" aria-labelledby="results-title">
                <div className="review-section-top"><div><p className="review-eyebrow">02 / READ THE EVIDENCE</p><h2 id="results-title">Your medication review</h2><p className="review-muted">{result.medications.length} selected concepts · {(result.processing_time_ms/1000).toFixed(1)}s server processing</p></div><button className="review-secondary" onClick={download}>Download review for discussion ↓</button></div>
                {reportText&&<div className="review-export" role="region" aria-label="Review text"><label htmlFor="review-export-text">Your review is ready</label><p>If the download did not start, select and copy the text below to save it.</p><textarea id="review-export-text" readOnly value={reportText} rows={8}/></div>}
                <div className="review-stats"><div><strong>{result.pairs_reviewed}</strong><span>Pairs compared</span></div><div><strong>{result.counts.label_mention||0}</strong><span>Direct label mentions</span></div><div><strong>{result.counts.ingredient_overlap||0}</strong><span>Possible ingredient overlaps</span></div><div><strong>{result.pairs_with_incomplete_sources}</strong><span>Pairs with incomplete sources</span></div></div>
                <p className="review-notice">These are evidence categories, not risk scores. <strong>“No direct mention” is never a no-interaction result.</strong> Read each excerpt in context, including passages that describe no effect.</p>
                <div className="review-filters"><label>Show <select value={filter} onChange={e=>{setFilter(e.target.value);setPage(0)}}><option value="all">All pairs ({result.pairs_reviewed})</option>{['ingredient_overlap','label_mention','incomplete_sources','no_direct_mention'].map(k=><option key={k} value={k}>{statusNames[k]} ({k==='incomplete_sources'?result.pairs_with_incomplete_sources:result.counts[k]||0})</option>)}</select></label><button className="review-secondary" onClick={()=>setMatrix(!matrix)}>{matrix?'Hide pair matrix':'Show pair matrix'}</button></div>
                {matrix&&<div className="review-matrix" tabIndex={0} aria-label="Scrollable pair evidence matrix"><table><caption>Click a cell to inspect that pair. Dots indicate no direct mention; they do not mean safe.</caption><thead><tr><th>Medication</th>{result.medications.map(m=><th key={m.rxcui}>{m.name}</th>)}</tr></thead><tbody>{result.medications.map((m,i)=><tr key={m.rxcui}><th>{m.name}</th>{result.medications.map((n,j)=>{const pair=result.pairs.find(p=>p.ids.includes(m.rxcui)&&p.ids.includes(n.rxcui));return <td key={n.rxcui}>{i===j?'—':pair&&<button onClick={()=>setFocusPair(pair)} className={pair.status} aria-label={`${m.name} and ${n.name}: ${statusNames[pair.status]}`}>{pair.status==='label_mention'?'L':pair.status==='ingredient_overlap'?'I':pair.status==='incomplete_sources'?'?':'·'}</button>}</td>})}</tr>)}</tbody></table><p>L = label mention · I = possible ingredient overlap · ? = incomplete sources · dot = no direct mention</p></div>}
                {focusPair&&<div><h3>Selected matrix pair</h3><PairCard pair={focusPair}/></div>}
                <div className="review-pairs">{pairs.slice(page*15,(page+1)*15).map(p=><PairCard key={p.ids.join('-')} pair={p}/>)}{!pairs.length&&<p>No pairs in this category. Check the other categories and source coverage.</p>}</div>
                {totalPages>1&&<div className="review-pagination"><button disabled={!page} onClick={()=>setPage(page-1)}>Previous</button><span>Page {page+1} of {totalPages}</span><button disabled={page+1>=totalPages} onClick={()=>setPage(page+1)}>Next</button></div>}
                <h3 className="review-sources-title">Source coverage, medicine by medicine</h3><p className="review-muted">One sample label is used per concept. It may differ from your exact strength, manufacturer, route or formulation. Confirm the original label against your package.</p>
                <div className="review-source-grid">{result.medications.map(m=><article className="review-source" key={m.rxcui}><h4>{m.name}</h4><p className="review-badge">{statusNames[m.status]}</p><p>RxCUI {m.rxcui} · {typeName[m.tty]}</p><p>Ingredient mapping: {m.ingredients.map(i=>i.name).join(', ')||'Unavailable'}{m.ingredient_resolution==='brand_or_concept_level'?' (concept-level; verify formulation)':''}</p>{m.labels.map(l=><div key={l.setid}><a href={l.url} target="_blank" rel="noreferrer">{l.title} ↗</a><p>Published {l.published_date} · version {l.version}</p><p>{m.label_matches} matching labels; one sampled.</p><p>{l.sections.map(s=>s.section).join(' · ')||'No relevant sections extracted'}{l.sections.some(s=>s.truncated)?' · Text was truncated':''}</p></div>)}</article>)}</div>
                <details className="review-method"><summary>What this review does—and what it cannot establish</summary><p>{result.method}</p><ul>{result.limitations.map(t=><li key={t}>{t}</li>)}</ul><p>Clinical validation has not been established. A pharmacist or prescriber can review your actual products, doses, medical history and full regimen.</p></details>
            </section>}
            {!result&&<section className="review-explainer"><article><span>01</span><h3>Find the right name</h3><p>Search ingredient and brand terminology. Confirm combination products and your exact formulation.</p></article><article><span>02</span><h3>Inspect the source</h3><p>Read original label passages with links and version information. Missing evidence stays visible.</p></article><article><span>03</span><h3>Bring better questions</h3><p>Download a review to discuss with a pharmacist. The tool never recommends a dose or medication change.</p></article></section>}
        </main>
        <footer className="review-footer"><p>Terminology: RxNorm · Label source: DailyMed · No account required</p><p>{catalog&&<>Catalog snapshot: {new Date(catalog.metadata.retrieved_at).toLocaleDateString()} · </>}Source responses may be cached for up to 12 hours.</p><p>This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.</p><a href="https://github.com/vajja1405/drug-interaction-rag-chatbot" target="_blank" rel="noreferrer">Source, methods & limitations ↗</a></footer>
    </div>
}
