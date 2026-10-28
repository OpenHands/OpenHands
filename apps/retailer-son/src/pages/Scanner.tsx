import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PRODUCTS } from '../data/products'

export default function Scanner() {
  const [barcode, setBarcode] = useState('')
  const [msg, setMsg] = useState<string | null>(null)
  const navigate = useNavigate()

  function lookup(b: string) {
    const found = PRODUCTS.find(p => p.barcode === b)
    if (found) navigate(`/product/${b}`)
    else setMsg('No product with that barcode in the test catalogue.')
  }

  return (
    <div className="page">
      <h2>Scanner</h2>
      <p>Camera scanning not implemented in this MVP. Use manual barcode entry below.</p>
      <div>
        <input aria-label="barcode" placeholder="Enter exact EAN-13 barcode" value={barcode} onChange={e => setBarcode(e.target.value.trim())} />
        <button onClick={() => lookup(barcode)}>Lookup</button>
      </div>
      {msg && <div className="error">{msg}</div>}
      <div>
        <button onClick={() => navigate('/')}>Back to Home</button>
      </div>
    </div>
  )
}
