// modelsInfo.js - Tracer-B7, BASNet, and RMBG-2.0
const ModelsInfo = {
    tracer: { 
        displayName: 'Tracer-B7', 
        shortName: "Tracer",
        sourceUrl: 'https://github.com/OPHoperHPO/image-background-remove-tool#%EF%B8%8F-how-does-it-work', 
        apiUrlVar: 'REACT_APP_TRACER_URL'
    },
    basnet: { 
        displayName: 'BASNet', 
        shortName: "BASNet",
        sourceUrl: 'https://github.com/OPHoperHPO/image-background-remove-tool#%EF%B8%8F-how-does-it-work', 
        apiUrlVar: 'REACT_APP_BASNET_URL'
    },
    rmbg: { 
        displayName: 'RMBG-2.0', 
        shortName: "RMBG",
        sourceUrl: 'https://huggingface.co/briaai/RMBG-2.0', 
        apiUrlVar: 'REACT_APP_RMBG_URL'
    }
};

export default ModelsInfo;
