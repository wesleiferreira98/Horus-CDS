import { useState, useEffect, useCallback, useRef } from 'react'
import Navbar from './components/Navbar'
import ServerConfig from './components/ServerConfig'
import StatsCards from './components/StatsCards'
import ChartsSection from './components/ChartsSection'
import LogTable from './components/LogTable'
import ModelModal from './components/ModelModal'
import ConfigModal from './components/ConfigModal'
import MetricsSummary from './components/MetricsSummary'
import SimulationConfig from './components/SimulationConfig'

const REFRESH_INTERVAL = 10000

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('horus-theme') || 'dark')
  const [apiBase, setApiBase] = useState(null)
  const [configLoading, setConfigLoading] = useState(true)
  const [monitoring, setMonitoring] = useState(false)
  const [currentModel, setCurrentModel] = useState('Horus-CDS V4 (TCN)')
  const [filter, setFilter] = useState('recentes')
  const [currentPage, setCurrentPage] = useState(1)
  const [chartType, setChartType] = useState('bar')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [connected, setConnected] = useState(false)
  const [apiStatus, setApiStatus] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [simApplying, setSimApplying] = useState(false)
  const [modelModalOpen, setModelModalOpen] = useState(false)
  const [configModalOpen, setConfigModalOpen] = useState(false)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('horus-theme', theme)
  }, [theme])

  useEffect(() => {
    fetch('/api-config')
      .then(r => r.json())
      .then(cfg => {
        setApiBase(cfg.apiUrl)
        setConfigLoading(false)
      })
      .catch(() => {
        setApiBase(`http://${window.location.hostname}:5000`)
        setConfigLoading(false)
      })
  }, [])

  const fetchData = useCallback(async () => {
    if (!apiBase) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${apiBase}/gerar_dados?filter=${filter}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      const json = await res.json()
      setData(json)
      setConnected(true)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err.message)
      setConnected(false)
    } finally {
      setLoading(false)
    }
  }, [apiBase, filter])

  const fetchStatus = useCallback(async () => {
    if (!apiBase) return
    try {
      const res = await fetch(`${apiBase}/status`)
      if (res.ok) {
        const json = await res.json()
        setApiStatus(json)
        setCurrentModel(json.active_model)
        setConnected(true)
      }
    } catch {}
  }, [apiBase])

  useEffect(() => {
    if (apiBase) fetchStatus()
  }, [apiBase, fetchStatus])

  useEffect(() => {
    if (monitoring) {
      fetchData()
      fetchStatus()
      intervalRef.current = setInterval(() => {
        fetchData()
        fetchStatus()
      }, REFRESH_INTERVAL)
    } else {
      clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [monitoring, fetchData, fetchStatus])

  const handleFilterChange = (f) => {
    setFilter(f)
    setCurrentPage(1)
  }

  const handleSelectModel = async (model) => {
    if (!apiBase) return
    setLoading(true)
    try {
      const res = await fetch(`${apiBase}/set_model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      })
      const json = await res.json()
      if (res.ok) {
        setCurrentModel(model)
        setData(null)
        setModelModalOpen(false)
        await fetchData()
      } else {
        setError(json.error || 'Erro ao alterar o modelo.')
      }
    } catch {
      setError('Erro ao conectar à API.')
    } finally {
      setLoading(false)
    }
  }

  const handleSimConfig = async (config) => {
    if (!apiBase) return
    setSimApplying(true)
    try {
      const res = await fetch(`${apiBase}/set_simulation_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const json = await res.json()
      if (!res.ok) setError(json.error || 'Erro ao configurar simulação.')
    } catch {
      setError('Erro ao conectar à API.')
    } finally {
      setSimApplying(false)
    }
  }

  const handleToggleMode = async (simulationMode) => {
    if (!apiBase) return
    try {
      const res = await fetch(`${apiBase}/toggle_mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulation_mode: simulationMode }),
      })
      const json = await res.json()
      if (!res.ok) setError(json.error || 'Erro ao alterar modo.')
      else await fetchStatus()
    } catch {}
  }

  if (configLoading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-12 h-12 border-4 border-white/20 border-t-cyan-400 rounded-full animate-spin mx-auto" />
          <p className="text-gray-400 text-sm">Conectando ao Hórus-CDS...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors duration-200">
      {loading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999] pointer-events-none">
          <div className="w-12 h-12 border-4 border-white/20 border-t-cyan-400 rounded-full animate-spin" />
        </div>
      )}

      <Navbar
        theme={theme}
        onThemeToggle={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
        currentModel={currentModel}
        monitoring={monitoring}
        onOpenModelModal={() => setModelModalOpen(true)}
        onOpenConfigModal={() => setConfigModalOpen(true)}
      />

      <main className="w-full px-4 sm:px-6 lg:px-8 py-6 space-y-5">
        {/* Header banner */}
        <header className="rounded-2xl overflow-hidden">
          <div className="bg-gradient-to-r from-blue-700 via-cyan-700 to-blue-800 dark:from-blue-950 dark:via-cyan-950 dark:to-blue-950 px-8 py-5 text-white border border-cyan-800/40 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Monitoramento em Tempo Real Hórus-CDS</h1>
              <p className="text-blue-200 text-sm mt-1">Detecção de intrusão em redes de smart grid</p>
            </div>
            {monitoring && (
              <div className="flex items-center gap-2.5 px-4 py-2 rounded-full bg-red-500/20 border border-red-400/50 backdrop-blur-sm shrink-0">
                <span className="w-2.5 h-2.5 rounded-full bg-red-400 animate-pulse" />
                <span className="text-red-300 text-sm font-bold tracking-widest uppercase">Ao Vivo</span>
              </div>
            )}
          </div>
        </header>

        <ServerConfig
          apiBase={apiBase}
          monitoring={monitoring}
          onToggleMonitor={() => setMonitoring(m => !m)}
          onSync={() => { fetchData(); fetchStatus() }}
          apiStatus={apiStatus}
          onToggleMode={handleToggleMode}
          loading={loading}
          connected={connected}
          lastUpdated={lastUpdated}
        />

        {error && (
          <div className="p-4 rounded-xl bg-red-900/20 border border-red-700/50 text-red-400 text-sm flex items-start gap-3">
            <span className="shrink-0">⚠</span>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="ml-auto text-red-600 hover:text-red-400 transition-colors shrink-0">✕</button>
          </div>
        )}

        {apiStatus?.simulation_mode === true && (
          <SimulationConfig onApply={handleSimConfig} applying={simApplying} />
        )}
        <MetricsSummary data={data} apiStatus={apiStatus} />
        <StatsCards data={data} />
        <ChartsSection data={data} chartType={chartType} />
        <LogTable
          data={data}
          currentPage={currentPage}
          onPageChange={setCurrentPage}
          filter={filter}
          onFilterChange={handleFilterChange}
        />
      </main>

      <footer className="mt-12 py-4 text-center text-xs text-gray-400 dark:text-gray-600 border-t border-gray-200 dark:border-gray-800">
        Hórus-CDS © 2024 — Todos os direitos reservados
      </footer>

      <ModelModal
        open={modelModalOpen}
        onClose={() => setModelModalOpen(false)}
        currentModel={currentModel}
        onSelect={handleSelectModel}
      />

      <ConfigModal
        open={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
        chartType={chartType}
        onSave={(type) => { setChartType(type); setConfigModalOpen(false) }}
      />
    </div>
  )
}
