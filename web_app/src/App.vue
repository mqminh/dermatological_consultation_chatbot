<template>
  <div class="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(59,130,246,0.18),_transparent_30%),linear-gradient(135deg,_#f8fbff_0%,_#eef5ff_45%,_#f9fbff_100%)] text-slate-900 font-sans">
    <div class="mx-auto flex max-w-5xl flex-col px-3 pb-40 pt-3 sm:px-6 sm:pt-6 lg:px-8">
      <div class="pointer-events-none absolute inset-x-0 top-0 h-48 overflow-hidden">
        <div class="mx-auto h-40 w-72 rounded-full bg-cyan-400/20 blur-3xl"></div>
      </div>

      <Header v-model:language="language" v-model:llmMode="llmMode" />

      <main class="relative z-10 mt-4 rounded-[24px] border border-white/70 bg-white/80 p-3 shadow-[0_20px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:mt-6 sm:rounded-[32px] sm:p-6 lg:p-8">
        <div v-if="messages.length === 0" class="flex flex-col items-center justify-center rounded-[20px] border border-dashed border-sky-200 bg-sky-50/70 px-4 py-12 text-center sm:px-6 sm:py-16">
          <div class="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 to-cyan-400 text-white shadow-lg shadow-sky-200 sm:h-16 sm:w-16">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7 sm:h-8 sm:w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M4 16l4.586-4.586a2 2 0 012.828 0L13 13m-1-8l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </div>
          <h2 class="text-xl font-semibold text-slate-800 sm:text-2xl">
            {{ language === 'Vi' ? 'Bắt đầu một buổi tư vấn da liễu' : 'Start a dermatology consultation' }}
          </h2>
          <p class="mt-2 max-w-2xl text-sm text-slate-600 sm:text-base">
            {{ language === 'Vi' ? 'Tải lên ảnh da liễu để nhận phân tích ban đầu và lời khuyên sơ bộ.' : 'Upload a skin image to get an initial analysis and preliminary guidance.' }}
          </p>
        </div>

        <ChatFeed :messages="messages" :lang="language" />

        <div v-if="isLoading" class="mt-6 flex items-center justify-center gap-3 rounded-2xl border border-sky-100 bg-sky-50/80 px-4 py-4 text-sm font-semibold text-sky-700">
          <svg class="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {{ language === 'Vi' ? 'Đang xử lý...' : 'Processing...' }}
        </div>
      </main>
    </div>

    <InputArea
        :isLoading="isLoading"
        :lang="language"
        :hasDiagnosis="currentDisease !== null"
        @send-image="handleSendImage"
        @send-text="handleSendText"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Header from './components/Header.vue'
import ChatFeed from './components/ChatFeed.vue'
import InputArea from './components/InputArea.vue'
import { sendConsultationRequest, sendFollowUpMessage } from './services/api'

const language = ref('En')
const llmMode = ref('local')
const messages = ref([])
const isLoading = ref(false)
const currentDisease = ref(null)

const smoothScroll = () => {
  setTimeout(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
  }, 100)
}

const handleSendImage = async ({ file, preview }) => {
  messages.value.push({ role: 'user', image: preview })
  isLoading.value = true
  smoothScroll()

  try {
    const data = await sendConsultationRequest(file, language.value, llmMode.value)
    if (data.success) {
      currentDisease.value = data.data.disease
      messages.value.push({
        role: 'model',
        text: data.data.consultation,
        disease: data.data.disease,
        confidence: data.data.confidence
      })
    } else {
      messages.value.push({ role: 'model', text: `**Error:** ${data.message}` })
    }
  } catch (error) {
    messages.value.push({ role: 'model', text: `**Connection Error:** Cannot communicate with the server.` })
  } finally {
    isLoading.value = false
    smoothScroll()
  }
}

const handleSendText = async (text) => {
  messages.value.push({ role: 'user', text: text })
  isLoading.value = true
  smoothScroll()

  const historyPayload = messages.value.map(msg => ({
    role: msg.role,
    text: msg.text
  }))

  try {
    const data = await sendFollowUpMessage(text, currentDisease.value, historyPayload, language.value, llmMode.value)
    if (data.success) {
      messages.value.push({ role: 'model', text: data.data.reply })
    } else {
      messages.value.push({ role: 'model', text: `**Error:** ${data.message}` })
    }
  } catch (error) {
    messages.value.push({ role: 'model', text: `**Connection Error:** Cannot communicate with the server.` })
  } finally {
    isLoading.value = false
    smoothScroll()
  }
}
</script>