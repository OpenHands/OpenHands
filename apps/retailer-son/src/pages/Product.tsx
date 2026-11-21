import React, { useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { PRODUCTS } from '../data/products'

export default function ProductPage() {
  const { barcode } = useParams<{ barcode: string }>()
  const navigate = useNavigate()
  const product = PRODUCTS.find(p => p.barcode === barcode)

  useEffect(() => {
    if (product) {
      const recent = JSON.parse(localStorage.getItem('recent') || '[]')
      recent.unshift({ barcode: product.barcode, name: product.name, scanned: false, time: Date.now() })
      localStorage.setItem('recent', JSON.stringify(recent.slice(0, 50)))
    }
  }, [product])

  if (!product) {
    return (
      <div className="page">
        <div>Product not found for barcode {barcode}</div>
        <button onClick={() => navigate(-1)}>Back</button>
      </div>
    )
  }

  return (
    <div className="page">
      <button onClick={() => navigate(-1)}>Back</button>
      <h2>{product.name}</h2>
      <div>{product.pack}</div>
      <div>Barcode: {product.barcode}</div>
      <div>RRP: {product.rrp}</div>
      <h3>Supplier prices (test data)</h3>
      <ul>
        {Object.entries(product.suppliers).map(([s, price]) => (
          <li key={s} style={{ fontWeight: s === product.lowest.supplier ? '700' : '400' }}>
            {s}: {price}{' '}
            {s === product.lowest.supplier ? <strong>(Lowest)</strong> : null}
          </li>
        ))}
      </ul>

      <div>
        <Link to="/scanner">Scan another product</Link>
      </div>
    </div>
  )
}
