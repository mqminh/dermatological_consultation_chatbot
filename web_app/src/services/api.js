const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:5001`

export const sendConsultationRequest = async (file, lang) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('lang', lang)

    const response = await fetch(`${API_BASE_URL}/api/consult`, {
        method: 'POST',
        body: formData
    })

    if (!response.ok) {
        throw new Error('Network response error')
    }

    return await response.json()
}

export const sendFollowUpMessage = async (message, disease, history, lang) => {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message, disease, history, lang })
    })

    if (!response.ok) {
        throw new Error('Network response error')
    }

    return await response.json()
}