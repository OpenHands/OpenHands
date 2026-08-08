declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV?: string;
      PUBLIC_URL?: string;
    }
  }
}

export {};
