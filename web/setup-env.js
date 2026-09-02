const fs = require('fs');
const targetPath = './src/environments/environment.ts';
const baseUrl = process.env.API_BASE_URL !== undefined && process.env.API_BASE_URL !== '' 
  ? process.env.API_BASE_URL 
  : 'http://localhost:8000';
const envConfigFile = `export const environment = {
  production: ${process.env.NODE_ENV === 'production'},
  apiBaseUrl: '${baseUrl}'
};`;
fs.writeFileSync(targetPath, envConfigFile);
console.log(`Generated ${targetPath} with API_BASE_URL = ${baseUrl}`);
