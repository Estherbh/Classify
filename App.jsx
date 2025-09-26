import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { MapPin, Download, BarChart3, Satellite, TreePine, Coffee, Palmtree, Moon, Sun } from 'lucide-react'
import './App.css'

function App() {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [selectedPolygon, setSelectedPolygon] = useState(null)
  const [classificationResults, setClassificationResults] = useState([])

  // Données simulées pour le dashboard
  const mockResults = [
    {
      id: 1,
      culture: 'palmier à huile',
      predicted_culture: 'palmier à huile',
      predicted_proba: 0.95,
      confidence_level: 5,
      recommended_action: 'Très haute confiance: Peut être utilisé pour la cartographie automatique.',
      coordinates: [5.5, -3.2]
    },
    {
      id: 2,
      culture: 'Cocoa',
      predicted_culture: 'palmier à huile',
      predicted_proba: 0.72,
      confidence_level: 4,
      recommended_action: 'Confiance élevée: Cartographie automatique recommandée, vérification ponctuelle possible.',
      coordinates: [5.6, -3.1]
    },
    {
      id: 3,
      culture: 'Café Arabica',
      predicted_culture: 'Café Arabica',
      predicted_proba: 0.88,
      confidence_level: 5,
      recommended_action: 'Très haute confiance: Peut être utilisé pour la cartographie automatique.',
      coordinates: [5.4, -3.3]
    },
    {
      id: 4,
      culture: 'palmier à huile',
      predicted_culture: 'Cocoa',
      predicted_proba: 0.45,
      confidence_level: 3,
      recommended_action: 'Confiance moyenne: Vérification visuelle recommandée si possible.',
      coordinates: [5.7, -3.0]
    }
  ]

  useEffect(() => {
    setClassificationResults(mockResults)
    // Appliquer le mode sombre par défaut
    document.documentElement.classList.add('dark')
  }, [])

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode)
    document.documentElement.classList.toggle('dark')
  }

  const getConfidenceColor = (level) => {
    const colors = {
      1: 'bg-red-500',
      2: 'bg-orange-500',
      3: 'bg-yellow-500',
      4: 'bg-green-400',
      5: 'bg-green-600'
    }
    return colors[level] || 'bg-gray-500'
  }

  const getConfidenceLabel = (level) => {
    const labels = {
      1: 'Très faible',
      2: 'Faible',
      3: 'Moyenne',
      4: 'Élevée',
      5: 'Très haute'
    }
    return labels[level] || 'Inconnue'
  }

  const getCultureIcon = (culture) => {
    if (culture.includes('palmier')) return <Palmtree className="h-4 w-4" />
    if (culture.includes('Cocoa') || culture.includes('cacao')) return <Coffee className="h-4 w-4" />
    if (culture.includes('Café')) return <Coffee className="h-4 w-4" />
    return <TreePine className="h-4 w-4" />
  }

  const exportResults = (format) => {
    // Simulation de l'export
    console.log(`Export des résultats au format ${format}`)
    alert(`Export ${format} en cours...`)
  }

  const totalPolygons = classificationResults.length
  const highConfidenceCount = classificationResults.filter(r => r.confidence_level >= 4).length
  const averageConfidence = classificationResults.reduce((acc, r) => acc + r.confidence_level, 0) / totalPolygons

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Satellite className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold">Classification Cultures Agricoles</h1>
          </div>
          <div className="flex items-center space-x-4">
            <Button
              variant="outline"
              size="icon"
              onClick={toggleDarkMode}
            >
              {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            <Button onClick={() => exportResults('CSV')}>
              <Download className="h-4 w-4 mr-2" />
              Exporter
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        {/* Statistiques générales */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Polygones</CardTitle>
              <MapPin className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{totalPolygons}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Haute Confiance</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{highConfidenceCount}</div>
              <p className="text-xs text-muted-foreground">
                {Math.round((highConfidenceCount / totalPolygons) * 100)}% du total
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Confiance Moyenne</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{averageConfidence.toFixed(1)}/5</div>
              <Progress value={(averageConfidence / 5) * 100} className="mt-2" />
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Précision Modèle</CardTitle>
              <Satellite className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">63.4%</div>
              <p className="text-xs text-muted-foreground">Random Forest</p>
            </CardContent>
          </Card>
        </div>

        {/* Contenu principal avec onglets */}
        <Tabs defaultValue="results" className="space-y-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="results">Résultats</TabsTrigger>
            <TabsTrigger value="map">Carte Interactive</TabsTrigger>
            <TabsTrigger value="analytics">Analyses</TabsTrigger>
          </TabsList>
          
          <TabsContent value="results" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Résultats de Classification</CardTitle>
                <CardDescription>
                  Classification des cultures agricoles avec niveaux de confiance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {classificationResults.map((result) => (
                    <div
                      key={result.id}
                      className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-accent/50 cursor-pointer transition-colors"
                      onClick={() => setSelectedPolygon(result)}
                    >
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {getCultureIcon(result.predicted_culture)}
                          <span className="font-medium">{result.predicted_culture}</span>
                        </div>
                        <Badge
                          className={`${getConfidenceColor(result.confidence_level)} text-white`}
                        >
                          {getConfidenceLabel(result.confidence_level)}
                        </Badge>
                      </div>
                      <div className="text-right">
                        <div className="font-bold">{Math.round(result.predicted_proba * 100)}%</div>
                        <div className="text-sm text-muted-foreground">
                          Réel: {result.culture}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="map" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Carte Interactive</CardTitle>
                <CardDescription>
                  Visualisation géospatiale des résultats de classification
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96 bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <MapPin className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">
                      Carte interactive en cours de développement
                    </p>
                    <p className="text-sm text-muted-foreground mt-2">
                      Intégration avec Leaflet/Folium prévue
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Distribution des Cultures</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {['palmier à huile', 'Cocoa', 'Café Arabica'].map((culture) => {
                      const count = classificationResults.filter(r => r.predicted_culture === culture).length
                      const percentage = (count / totalPolygons) * 100
                      return (
                        <div key={culture} className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            {getCultureIcon(culture)}
                            <span className="text-sm">{culture}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Progress value={percentage} className="w-20" />
                            <span className="text-sm font-medium">{count}</span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Niveaux de Confiance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {[5, 4, 3, 2, 1].map((level) => {
                      const count = classificationResults.filter(r => r.confidence_level === level).length
                      const percentage = (count / totalPolygons) * 100
                      return (
                        <div key={level} className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <div className={`w-3 h-3 rounded-full ${getConfidenceColor(level)}`}></div>
                            <span className="text-sm">{getConfidenceLabel(level)}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Progress value={percentage} className="w-20" />
                            <span className="text-sm font-medium">{count}</span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        {/* Détails du polygone sélectionné */}
        {selectedPolygon && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Détails du Polygone #{selectedPolygon.id}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">Classification</h4>
                  <p><strong>Culture réelle:</strong> {selectedPolygon.culture}</p>
                  <p><strong>Prédiction:</strong> {selectedPolygon.predicted_culture}</p>
                  <p><strong>Probabilité:</strong> {Math.round(selectedPolygon.predicted_proba * 100)}%</p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Confiance</h4>
                  <div className="flex items-center space-x-2 mb-2">
                    <Badge className={`${getConfidenceColor(selectedPolygon.confidence_level)} text-white`}>
                      Niveau {selectedPolygon.confidence_level}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {selectedPolygon.recommended_action}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export default App


