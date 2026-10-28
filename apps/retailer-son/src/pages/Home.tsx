import React, { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { PRODUCTS } from '../data/products'

export default function Home() {
  const [q, setQ] = useState('')
  const results = useMemo(() => {
    const t = q.trim().toLowerCase()
    if (!t) return PRODUCTS
    return PRODUCTS.filter(
      p => p.name.toLowerCase().includes(t) || p.barcode === t
    )
  }, [q])

  return (
    <div className="page">
      <div className="search">
        <input aria-label="search" placeholder="Search by name or exact barcode" value={q} onChange={e => setQ(e.target.value)} />
      </div>

      {results.length === 0 ? (
        <div className="empty">No products found.</div>
      ) : (
        <div className="grid">
          {results.map(p => (
            <Link key={p.id} to={`/product/${p.barcode}`} className="card" role="button">
              <h3>{p.name}</h3>
              <div>{p.pack}</div>
              <div>Barcode: {p.barcode}</div>
              <div>RRP: {p.rrp}</div>
              <div>Lowest: {p.lowest.supplier}, {p.lowest.price}</div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
