export const sendConsultationRequest = async (file, lang) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('lang', lang)

    const response = await fetch('http://127.0.0.1:5000/api/consult', {
        method: 'POST',
        body: formData
    })

    if (!response.ok) {
        throw new Error('Network response error')
    }

    return await response.json()
}