Retailer Son - MVP

This is a minimal Vite + React TypeScript scaffold implementing a 3-product MVP:
- Home: product list + search
- Scanner: manual barcode entry (camera scanning requires device + HTTPS)
- Recent: recent lookups stored in localStorage
- Product details page with supplier comparison

To run:
1. cd apps/retailer-son
2. npm ci
3. npm run dev

Limitations:
- Camera scanning not implemented; manual entry provided.
- Uses localStorage for Recent.
