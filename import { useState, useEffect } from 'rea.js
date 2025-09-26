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