<template>
  <header class="sticky top-3 z-20 mb-4 flex flex-wrap items-center justify-between rounded-[24px] border border-white/70 bg-white/80 px-4 py-4 shadow-[0_10px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:px-6">
    <div>
      <div class="flex items-center gap-3">
        <div class="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 to-cyan-400 text-white shadow-lg shadow-sky-200">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div>
          <h1 class="text-xl font-semibold tracking-tight text-slate-800">Dermatology AI</h1>
          <p class="text-sm text-slate-500">Smart skin consultation assistant</p>
        </div>
      </div>
    </div>

    <select
        :value="modelValue"
        @change="$emit('update:modelValue', $event.target.value)"
        class="cursor-pointer rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
    >
      <option value="En">English</option>
      <option value="Vi">Tiếng Việt</option>
    </select>
  </header>
</template>

<script setup>
defineProps({
  modelValue: String
})
defineEmits(['update:modelValue'])
</script>