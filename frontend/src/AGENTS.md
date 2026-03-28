# AGENTS.md — Frontend

React 18 + TypeScript SPA：A modern web interface for multimodal RAG。

## Structure

```
frontend/src/
├── main.tsx              # React Application entrance
├── App.tsx               # Root component, routing configuration
├── pages/
│   ├── QueryPage.tsx     # Main query interface (immersive chat）
│   ├── PapersPage.tsx    # Paper list
│   ├── PaperDetailPage.tsx
│   └── PaperReaderPage.tsx # PDF reader
├── components/
│   ├── query/
│   │   ├── ThoughtProcess.tsx    # Foldable thought process
│   │   └── ConversationSidebar.tsx
│   ├── reader/
│   │   ├── PDFViewer.tsx         # PDF rendering
│   │   ├── ChatPanel.tsx         # In-reader chat
│   │   ├── TocSidebar.tsx        # Directory navigation
│   │   └── SelectionToolbar.tsx  # Text selection toolbar
│   ├── layout/
│   │   └── Header.tsx
│   └── ui/               # shadcn/ui components
├── stores/
│   ├── conversation-store.ts  # Zustand Status management
│   └── theme-store.ts
├── lib/
│   ├── api.ts            # API client
│   └── utils.ts
└── index.css
```

## Key Patterns

### Component Conventions
- Functional components with explicit return types: `function Foo(): JSX.Element`
- PascalCase file names in `components/` or `pages/`
- Props destructuring in function signature

### State Management
```typescript
// Zustand store
export const useConversationStore = create<ConversationStore>((set, get) => ({
  conversations: [],
  // ...
}))

// Custom hooks: useThemeStore.ts in hooks/
```

### Server State
- use `@tanstack/react-query` Perform server status management
- API calls are encapsulated in `lib/api.ts`

### LaTeX Support
- KaTeX Used for mathematical formula rendering
- `rehype-katex` + `remark-math` plug-in

### PDF Integration
- `@react-pdf-viewer` for PDF rendering
- Support jumping to reference page

## Tech Stack

- React 19, TypeScript ~5.9
- Tailwind CSS 4.x, shadcn/ui
- Zustand 5.x (Status management)
- React Query 5.x (Server status)
- KaTeX (LaTeX rendering)
- Vite 8.x (Build tools)

## Commands

```bash
npm run dev     # http://localhost:5173
npm run build   # Production build
npm run lint    # ESLint
```
