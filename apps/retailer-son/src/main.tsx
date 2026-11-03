import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import App from './App'
import Home from './pages/Home'
import ProductPage from './pages/Product'
import Scanner from './pages/Scanner'
import Recent from './pages/Recent'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Home />} />
          <Route path="product/:barcode" element={<ProductPage />} />
          <Route path="scanner" element={<Scanner />} />
          <Route path="recent" element={<Recent />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
