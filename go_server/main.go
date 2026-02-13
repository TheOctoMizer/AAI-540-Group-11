package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type HealthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
	Uptime  string `json:"uptime"`
}

type PingResponse struct {
	Message   string `json:"message"`
	Timestamp string `json:"timestamp"`
}

var startTime time.Time

func init() {
	startTime = time.Now()
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	uptime := time.Since(startTime).String()

	response := HealthResponse{
		Status:  "healthy",
		Service: "Go Target Server",
		Uptime:  uptime,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	log.Printf("Health check from %s", r.RemoteAddr)
}

func pingHandler(w http.ResponseWriter, r *http.Request) {
	response := PingResponse{
		Message:   "pong",
		Timestamp: time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	log.Printf("Ping from %s", r.RemoteAddr)
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/ping", pingHandler)

	addr := fmt.Sprintf(":%s", port)

	log.Println(strings.Repeat("=", 60))
	log.Printf("Go Target Server Starting")
	log.Println(strings.Repeat("=", 60))
	log.Printf("Listening on %s", addr)
	log.Printf("Endpoints:")
	log.Printf("  GET /health - Health check")
	log.Printf("  GET /ping   - Ping endpoint")
	log.Println(strings.Repeat("=", 60))

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
