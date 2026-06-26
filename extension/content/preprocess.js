/**
 * Preprocessing module — replicates the training pipeline from AdvancedPreprocessing.ipynb
 * so that text sent to the HF API matches what the model expects.
 */

// --- Emoji to text mapping (subset of common emojis in judol comments) ---
const EMOJI_MAP = {
  '🔥': 'fire',
  '😍': 'smiling_face_with_heart_eyes',
  '🥺': 'pleading_face',
  '😂': 'laughing',
  '👍': 'thumbs_up',
  '👎': 'thumbs_down',
  '❤️': 'red_heart',
  '💕': 'two_hearts',
  '💰': 'money_bag',
  '🎰': 'slot_machine',
  '🎲': 'dice',
  '💎': 'gem',
  '✌️': 'victory_hand',
  '😅': 'grinning_face_with_sweat',
  '👌': 'ok_hand',
  '💪': 'muscle',
  '🤑': 'money_mouth_face',
  '💵': 'dollar_bill',
  '🪙': 'coin',
  '💸': 'money_with_wings',
  '🏆': 'trophy',
  '⚠️': 'warning',
  '🎉': 'party_popper',
  '💯': 'hundred_points',
  '👀': 'eyes',
  '😊': 'smiling_face_with_smiling_eyes',
  '🤣': 'rolling_on_the_floor_laughing',
  '😎': 'smiling_face_with_sunglasses',
  '🙏': 'pray',
  '✨': 'sparkles',
  '🃏': 'joker',
  '🂡': 'ace_of_spades',
};

// --- Indonesian slang/judol normalization dictionary ---
// Covers saka-nlp equivalents + judol-specific terms
const SLANG_MAP = {
  // Common informal → formal
  'wd': 'withdraw',
  'depo': 'deposit',
  'gacor': 'gacor',
  'slot': 'slot',
  'maxwin': 'maxwin',
  'rtp': 'rtp',
  'promo': 'promo',
  'bonus': 'bonus',
  'jackpot': 'jackpot',
  'freebet': 'freebet',
  'mindepo': 'mindepo',
  'afk': 'afk',
  'wdk': 'withdraw',
  'wdp': 'withdraw',
  'rng': 'rng',
  'sbobet': 'sbobet',
  'judi': 'judi',
  'bet': 'bet',
  'togel': 'togel',
  'casino': 'casino',
  'poker': 'poker',
  'taruhan': 'taruhan',
  'bookie': 'bookie',
  // Informal Indonesian
  'ga': 'tidak',
  'gak': 'tidak',
  'gk': 'tidak',
  'tdk': 'tidak',
  'yg': 'yang',
  'udah': 'sudah',
  'sdh': 'sudah',
  'dah': 'sudah',
  'abis': 'habis',
  'bs': 'bisa',
  'bgt': 'banget',
  'bnget': 'banget',
  'bngett': 'banget',
  'bngettt': 'banget',
  'emg': 'emang',
  'emang': 'emang',
  'smg': 'semoga',
  'smg': 'semoga',
  'gw': 'saya',
  'gua': 'saya',
  'gue': 'saya',
  'lu': 'kamu',
  'kmu': 'kamu',
  'km': 'kamu',
  'org': 'orang',
  'dr': 'dari',
  'dpt': 'dapat',
  'dtg': 'datang',
  'dlm': 'dalam',
  'dsb': 'dan_sebagainya',
  'dll': 'dan_lain_lain',
  'tp': 'tapi',
  'tpi': 'tapi',
  'krn': 'karena',
  'krn': 'karena',
  'jd': 'jadi',
  'jd': 'jadi',
  'aj': 'saja',
  'sj': 'saja',
  'mah': 'saja',
  'doang': 'saja',
  'kek': 'seperti',
  'kyk': 'seperti',
  'kayak': 'seperti',
  'kayaknya': 'sepertinya',
  'trs': 'terus',
  'trus': 'terus',
  'mls': 'malas',
  'mlu': 'malu',
  'bnr': 'benar',
  'bner': 'benar',
  'brp': 'berapa',
  'btw': 'btw',
  'cmn': 'cuman',
  'cuman': 'cuman',
  'bkn': 'bukan',
  'bgtu': 'begitu',
  'gtu': 'begitu',
  'gtu': 'begitu',
  'dmn': 'dimana',
  'dmn': 'dimana',
  'klo': 'kalau',
  'kalo': 'kalau',
  'kl': 'kalau',
  'klau': 'kalau',
  'bsa': 'bisa',
  'bsk': 'besok',
  'bsok': 'besok',
  'msh': 'masih',
  'msih': 'masih',
  'dgn': 'dengan',
  'dg': 'dengan',
  'dg': 'dengan',
  'sbg': 'sebagai',
  'spt': 'seperti',
  'ap': 'apa',
  'apk': 'aplikasi',
  'utk': 'untuk',
  'bk': 'bukan',
  'skrg': 'sekarang',
  'skrng': 'sekarang',
  'mw': 'mau',
  'mao': 'mau',
  'mo': 'mau',
  'mlm': 'malam',
  'pgi': 'pagi',
  'siang': 'siang',
  'sore': 'sore',
  'ml': 'malam',
  'pg': 'pagi',
  'sg': 'singapore',
  'hk': 'hongkong',
  'main': 'main',
  'maen': 'main',
  'mna': 'main',
  'dp': 'deposit',
  'wd': 'withdraw',
  'sultan': 'sultan',
  'hoki': 'hoki',
  'luck': 'luck',
  'lucky': 'lucky',
  'win': 'win',
  'kalah': 'kalah',
  'lose': 'lose',
  'cheat': 'cheat',
  'hack': 'hack',
  'bot': 'bot',
  'admin': 'admin',
  'bo': 'bandar_online',
  'bandar': 'bandar',
  'link': 'link',
  'bio': 'bio',
  'referral': 'referral',
  'ref': 'referral',
  'invite': 'invite',
  'join': 'join',
  'daftar': 'daftar',
  'register': 'register',
  'reg': 'register',
  'claim': 'claim',
  'klaim': 'claim',
  'voucher': 'voucher',
  'cashback': 'cashback',
  'rebate': 'rebate',
  'rolling': 'rolling',
  'turnover': 'turnover',
  'to': 'turnover',
  'min': 'minimal',
  'maks': 'maksimal',
  'max': 'maksimal',
  'livechat': 'livechat',
  'cs': 'customer_service',
  'wa': 'whatsapp',
  'telegram': 'telegram',
  'tg': 'telegram',
  'ig': 'instagram',
  'fb': 'facebook',
  'tw': 'twitter',
  'yt': 'youtube',
  'tutor': 'tutorial',
  'trik': 'trik',
  'tips': 'tips',
  'hack': 'hack',
  'cheat': 'cheat',
  'scam': 'scam',
  'penipu': 'penipu',
  'tipu': 'tipu',
  'tipu': 'tipu',
  'trust': 'trust',
  'trusted': 'trusted',
  'legit': 'legit',
  'aman': 'aman',
  'safe': 'safe',
  'resmi': 'resmi',
  'official': 'official',
  'legal': 'legal',
  'izin': 'izin',
  'lisensi': 'lisensi',
  'license': 'license',
  'verifikasi': 'verifikasi',
  'verify': 'verify',
  'kk': 'kartu_keluarga',
  'ktp': 'ktp',
  'nik': 'nik',
  'rekening': 'rekening',
  'rek': 'rekening',
  'bank': 'bank',
  'transfer': 'transfer',
  'tf': 'transfer',
  'pulsa': 'pulsa',
  'ewallet': 'ewallet',
  'e-wallet': 'ewallet',
  'dana': 'dana',
  'ovo': 'ovo',
  'gopay': 'gopay',
  'shopeepay': 'shopeepay',
  'qris': 'qris',
  'crypto': 'crypto',
  'btc': 'bitcoin',
  'eth': 'ethereum',
  'usdt': 'usdt',
  'binance': 'binance',
};

/**
 * Step 1: Convert emojis to text labels (JS equivalent of emoji.demojize)
 */
function demojize(text) {
  for (const [emoji, label] of Object.entries(EMOJI_MAP)) {
    text = text.replace(new RegExp(emoji, 'g'), ` ${label} `);
  }
  // Fallback: replace any remaining emoji with space (strip non-ASCII symbols)
  text = text.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{200D}]/gu, ' ');
  return text;
}

/**
 * Step 2: Unicode normalization — convert bold/italic styled text to normal
 * (NFKD equivalent: decompose characters then strip non-ASCII)
 */
function normalizeUnicode(text) {
  // Normalize using NFKD-like approach
  // Maps mathematical bold/italic Latin letters to their normal equivalents
  const BOLD_MAP = {
    '𝐀':'A','𝐁':'B','𝐂':'C','𝐃':'D','𝐄':'E','𝐅':'F','𝐆':'G','𝐇':'H','𝐈':'I','𝐉':'J',
    '𝐊':'K','𝐋':'L','𝐌':'M','𝐍':'N','𝐎':'O','𝐏':'P','𝐐':'Q','𝐑':'R','𝐒':'S','𝐓':'T',
    '𝐔':'U','𝐕':'V','𝐖':'W','𝐗':'X','𝐘':'Y','𝐙':'Z',
    '𝐚':'a','𝐛':'b','𝐜':'c','𝐝':'d','𝐞':'e','𝐟':'f','𝐠':'g','𝐡':'h','𝐢':'i','𝐣':'j',
    '𝐤':'k','𝐥':'l','𝐦':'m','𝐧':'n','𝐨':'o','𝐩':'p','𝐪':'q','𝐫':'r','𝐬':'s','𝐭':'t',
    '𝐮':'u','𝐯':'v','𝐰':'w','𝐱':'x','𝐲':'y','𝐳':'z',
    '𝟎':'0','𝟏':'1','𝟐':'2','𝟑':'3','𝟒':'4','𝟓':'5','𝟔':'6','𝟕':'7','𝟖':'8','𝟗':'9',
  };
  const ITALIC_MAP = {
    '𝘈':'A','𝘉':'B','𝘊':'C','𝘋':'D','𝘈':'E','𝘍':'F','𝘎':'G','𝘏':'H','𝘐':'I','𝘑':'J',
    '𝘒':'K','𝘓':'L','𝘔':'M','𝘕':'N','𝘖':'O','𝘗':'P','𝘘':'Q','𝘙':'R','𝘚':'S','𝘛':'T',
    '𝘜':'U','𝘝':'V','𝘞':'W','𝘟':'X','𝘠':'Y','𝘡':'Z',
    '𝘢':'a','𝘣':'b','𝘤':'c','𝘥':'d','𝘦':'e','𝘧':'f','𝘨':'g','𝘩':'h','𝘪':'i','𝘫':'j',
    '𝘬':'k','𝘭':'l','𝘮':'m','𝘯':'n','𝘰':'o','𝘱':'p','𝘲':'q','𝘳':'r','𝘴':'s','𝘵':'t',
    '𝘶':'u','𝘷':'v','𝘸':'w','𝘹':'x','𝘺':'y','𝘻':'z',
  };
  const BOLD_ITALIC_MAP = {
    '𝙖':'A','𝙗':'b','𝙘':'c','𝙙':'d','𝙚':'e','𝙛':'f','𝙜':'g','𝙝':'h','𝙞':'i','𝙟':'j',
    '𝙠':'k','𝙡':'l','𝙢':'m','𝙣':'n','𝙤':'o','𝙥':'p','𝙦':'q','𝙧':'r','𝙨':'s','𝙩':'t',
    '𝙪':'u','𝙫':'v','𝙬':'w','𝙭':'x','𝙮':'y','𝙯':'z',
    '𝙰':'A','𝙱':'B','𝙲':'C','𝙳':'D','𝙴':'E','𝙵':'F','𝙶':'G','𝙷':'H','𝙸':'I','𝙹':'J',
    '𝙺':'K','𝙻':'L','𝙼':'M','𝙽':'N','𝙾':'O','𝙿':'P','𝚀':'Q','𝚁':'R','𝚂':'S','𝚃':'T',
    '𝚄':'U','𝚅':'V','𝚆':'W','𝚇':'X','𝚈':'Y','𝚉':'Z',
    '𝟘':'0','𝟙':'1','𝟚':'2','𝟛':'3','𝟜':'4','𝟝':'5','𝟞':'6','𝟟':'7','𝟠':'8','𝟡':'9',
  };
  // FRACTUR and DOUBLE-STRUCK (less common but seen in judol)
  const FRACTUR_MAP = {
    '𝔄':'A','𝔅':'B','𝔇':'D','𝔈':'E','𝔉':'F','𝔊':'G','𝔋':'P','𝔏':'L','𝔐':'M','𝔑':'N',
    '𝔒':'O','𝔔':'Q','𝔕':'S','𝔗':'T','𝔘':'U','𝔙':'V','𝔚':'W','𝔛':'X','𝔜':'Y','ℜ':'R',
    '𝔞':'a','𝔟':'b','𝔠':'c','𝔡':'d','𝔢':'e','𝔣':'f','𝔤':'g','𝔥':'h','𝔦':'i','𝔧':'j',
    '𝔨':'k','𝔩':'l','𝔪':'m','𝔫':'n','𝔬':'o','𝔭':'p','𝔮':'q','𝔯':'r','𝔰':'s','𝔱':'t',
    '𝔲':'u','𝔳':'v','𝔴':'w','𝔵':'x','𝔶':'y','𝔷':'z',
  };

  const allMaps = [BOLD_MAP, ITALIC_MAP, BOLD_ITALIC_MAP, FRACTUR_MAP];
  for (const map of allMaps) {
    for (const [styled, normal] of Object.entries(map)) {
      text = text.replace(new RegExp(styled, 'g'), normal);
    }
  }

  // Strip any remaining non-ASCII non-printable chars
  text = text.replace(/[^\x00-\x7F]/g, m => {
    // Keep common Indonesian chars (none needed since Indonesian uses Latin alphabet)
    return m.charCodeAt(0) > 127 ? ' ' : m;
  });
  return text;
}

/**
 * Step 3: URL masking — replace URLs with [URL]
 */
function maskUrls(text) {
  text = text.replace(/https?:\/\/\S+/gi, '[URL]');
  text = text.replace(/www\.\S+/gi, '[URL]');
  text = text.replace(/bit\s?[.,]\s?ly\/\w+/gi, '[URL]');
  // Handle [dot] obfuscation used by judol promoters
  text = text.replace(/\w+\s?\[dot\]\s?\w+/gi, '[URL]');
  // Handle dot-obfuscation like "com dot" patterns
  text = text.replace(/\w+\s?\.\s?(com|net|org|id|io|co)\b/gi, '[URL]');
  return text;
}

/**
 * Step 4: Lowercase
 */
function toLowercase(text) {
  return text.toLowerCase();
}

/**
 * Step 5: Clean symbols — keep alphanumeric, [URL], underscore, colon (from demojize)
 */
function cleanSymbols(text) {
  return text.replace(/[^a-zA-Z0-9\s\[\]\_:]/g, ' ');
}

/**
 * Step 6: Indonesian slang normalization (JS equivalent of saka-nlp)
 */
function normalizeSlang(text) {
  const words = text.split(/\s+/);
  const normalized = words.map(word => SLANG_MAP[word] || word);
  return normalized.join(' ');
}

/**
 * Step 7: Reduce repeated characters (e.g., "bngettt" → "bngett")
 * Reduces 3+ consecutive same chars down to 2
 */
function reduceRepeatedChars(text) {
  return text.replace(/(.)\1{2,}/g, '$1$1');
}

/**
 * Step 8: Final whitespace cleanup
 */
function cleanWhitespace(text) {
  return text.replace(/\s+/g, ' ').trim();
}

/**
 * Full preprocessing pipeline — mirrors AdvancedPreprocessing.ipynb
 */
function preprocessText(text) {
  if (!text || typeof text !== 'string') return '';
  let result = text;
  result = demojize(result);
  result = normalizeUnicode(result);
  result = maskUrls(result);
  result = toLowercase(result);
  result = cleanSymbols(result);
  result = normalizeSlang(result);
  result = reduceRepeatedChars(result);
  result = cleanWhitespace(result);
  return result;
}
