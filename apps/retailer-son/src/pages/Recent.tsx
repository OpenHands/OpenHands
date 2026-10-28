import React from 'react'
import { Link } from 'react-router-dom'

export default function Recent() {
  const recent = JSON.parse(localStorage.getItem('recent') || '[]') as Array<any>

  if (!recent || recent.length === 0) {
    return <div className="page">No recent lookups yet.</div>
  }

  return (
    <div className="page">
      <h2>Recent</h2>
      <ul>
        {recent.map((r, i) => (
          <li key={i}>
            <Link to={`/product/${r.barcode}`}>{r.name || r.barcode}</Link>
            <div style={{ fontSize: 12 }}>{new Date(r.time).toLocaleString()}</div>
            <div>{r.scanned ? 'scanned' : 'manual'}</div>
          </li>
        ))}
      </ul>
    </div>
  )
}
