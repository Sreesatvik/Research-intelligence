import { lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import './styles/components.css'
import AppShell from './app/AppShell'
import { AuthProvider } from './auth/AuthProvider'
import RequireAuth from './auth/RequireAuth'
import { LitDocumentsProvider } from './state/litDocuments'
import WorldLayout from './world/WorldLayout'

const HomePage = lazy(() => import('./pages/HomePage'))
const ChatPage = lazy(() => import('./pages/ChatPage'))
const LibraryPage = lazy(() => import('./pages/library/LibraryPage'))
const MapPage = lazy(() => import('./pages/MapPage'))
const LoginPage = lazy(() => import('./pages/LoginPage'))

// Every page renders inside WorldLayout, which owns the one persistent 3D canvas.
export default function App() {
  return (
    <LitDocumentsProvider>
      <AuthProvider>
        <Routes>
          <Route element={<WorldLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route element={<AppShell />}>
                <Route path="/chat/:chatId?" element={<ChatPage />} />
                <Route path="/library" element={<LibraryPage />} />
              </Route>
              <Route path="/map" element={<MapPage />} />
            </Route>
            <Route path="/ask" element={<Navigate to="/chat" replace />} />
            <Route path="/research" element={<Navigate to="/chat" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AuthProvider>
    </LitDocumentsProvider>
  )
}
