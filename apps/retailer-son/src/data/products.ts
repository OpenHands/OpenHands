export type Supplier = 'Wholesale A' | 'Wholesale B' | 'Wholesale C' | 'Wholesale D'

export type Product = {
  id: string
  name: string
  pack: string
  barcode: string
  rrp: string
  casePack?: string
  suppliers: Record<Supplier, string>
  lowest: { supplier: Supplier; price: string }
}

export const PRODUCTS: Product[] = [
  {
    id: 'p1',
    name: 'Bulk Vitamin B12 1000µg',
    pack: '60 tablets',
    barcode: '5060343746695',
    rrp: '£9.99',
    casePack: '1 × 10',
    suppliers: {
      'Wholesale A': '£70.00',
      'Wholesale B': '£54.00',
      'Wholesale C': '£80.00',
      'Wholesale D': '£70.00'
    },
    lowest: { supplier: 'Wholesale B', price: '£54.00' }
  },
  {
    id: 'p2',
    name: 'Lyca Mobile SIM',
    pack: 'SIM card',
    barcode: '5060209561462',
    rrp: '£1.49',
    casePack: '1 × 1',
    suppliers: {
      'Wholesale A': '£0.50',
      'Wholesale B': '£0.70',
      'Wholesale C': '£0.70',
      'Wholesale D': '£0.60'
    },
    lowest: { supplier: 'Wholesale A', price: '£0.50' }
  },
  {
    id: 'p3',
    name: "Johnson's Baby Oil 200ml",
    pack: '200 ml bottle',
    barcode: '3574669909594',
    rrp: '£2.20',
    casePack: '1 × 10',
    suppliers: {
      'Wholesale A': '£15.00',
      'Wholesale B': '£16.00',
      'Wholesale C': '£15.50',
      'Wholesale D': '£16.70'
    },
    lowest: { supplier: 'Wholesale A', price: '£15.00' }
  }
]
