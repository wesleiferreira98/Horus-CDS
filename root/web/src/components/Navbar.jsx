function parseModel(name) {
  const match = name.match(/V(\d+)\s*\((\w+)\)/)
  if (match) return `V${match[1]} · ${match[2]}`
  return name
}

function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M12 7a5 5 0 100 10 5 5 0 000-10z" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
    </svg>
  )
}

export default function Navbar({ theme, onThemeToggle, currentModel, monitoring, onOpenModelModal, onOpenConfigModal }) {
  const modelBadge = parseModel(currentModel)

  return (
    <nav className="sticky top-0 z-40 bg-white dark:bg-gray-900 border-b border-brand-muted/30 dark:border-gray-800 shadow-sm">
      <div className="w-full px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3 min-w-0">
          <img src="/images/logoSPTI.png" alt="Hórus" className="h-8 w-8 object-contain shrink-0" />
          <span className="font-bold text-brand-primary dark:text-white text-lg tracking-tight hidden sm:block">Hórus-CDS</span>

          <button
            onClick={onOpenModelModal}
            title="Trocar modelo"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold
                       bg-brand-bg dark:bg-brand-primary/20 text-brand-primary dark:text-white/80
                       border border-brand-primary/30 dark:border-brand-primary/50
                       hover:bg-brand-primary/10 dark:hover:bg-brand-primary/30 transition-colors cursor-pointer"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-brand-primary dark:bg-white/80 animate-pulse shrink-0" />
            <span className="truncate">{modelBadge}</span>
          </button>

          {monitoring && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold
                            bg-red-50 dark:bg-red-500/15 text-red-500 dark:text-red-400
                            border border-red-200 dark:border-red-500/40">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 dark:bg-red-400 animate-pulse" />
              AO VIVO
            </div>
          )}
        </div>

        {/* Right controls */}
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            onClick={onThemeToggle}
            className="p-2 rounded-lg text-brand-muted hover:text-brand-primary hover:bg-brand-bg
                       dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-700 transition-colors"
            title={theme === 'dark' ? 'Mudar para modo claro' : 'Mudar para modo escuro'}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>

          <button
            onClick={() => window.print()}
            className="hidden sm:block px-3 py-1.5 rounded-lg text-sm text-brand-soft dark:text-gray-300
                       hover:text-brand-primary dark:hover:text-white hover:bg-brand-bg dark:hover:bg-gray-700 transition-colors"
          >
            Relatório
          </button>

          <button
            onClick={onOpenConfigModal}
            className="px-3 py-1.5 rounded-lg text-sm text-brand-soft dark:text-gray-300
                       hover:text-brand-primary dark:hover:text-white hover:bg-brand-bg dark:hover:bg-gray-700 transition-colors"
          >
            Config
          </button>

          <button className="hidden sm:block px-3 py-1.5 rounded-lg text-sm text-brand-soft dark:text-gray-300
                             hover:text-brand-primary dark:hover:text-white hover:bg-brand-bg dark:hover:bg-gray-700 transition-colors">
            Sobre
          </button>
        </div>
      </div>
    </nav>
  )
}
