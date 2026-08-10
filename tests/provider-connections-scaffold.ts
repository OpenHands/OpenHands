/**
 * Provider Connections tests (scaffold, draft).
 *
 * MSW mocks + vitest + playwright e2e covering:
 *  - Connect-a-provider wizard (vendor/key -> test -> pick -> confirm)
 *  - Provider Connections section (refresh / rotate / disconnect)
 *  - secret-by-name profile resolution (profile references connection key,
 *    never inline-duplicates it)
 *
 * e2e must mock provider /models + validate; never hit live vendors.
 *
 * Tracking: OpenHands/OpenHands#15492, Linear OSS-5295.
 * Scope: OpenHands frontend PR5 of the provider-connections plan.
 *
 * TODO (implementation): depends on PR3 (wizard) + PR4 (connections section).
 */

export {};
