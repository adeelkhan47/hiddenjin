import streamlit as st
import pandas as pd
import asyncio
import aiodns
from email_validator import validate_email, EmailNotValidError
from io import StringIO
import logging
import time

# ------------- CONFIG -----------------
st.set_page_config(page_title="HIDDENJIN Email Processor", layout="wide")
st.title("📧 HIDDENJIN Email Processor")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

banned_keywords = [
    "no-reply", "noreply", "donotreply", "do_not_reply", "bounce", "null", "invalid",
    "undisclosed-recipients", "abuse", "spam", "postmaster", "webmaster", "hostmaster",
    "dmca", "legal", "mailer-daemon", "mailerdaemon", "autoreply", "auto_reply", "daemon",
    "nobody", "support", "help", "info", "information", "contact", "service"
]

# Expanded provider dictionary with regional and subdomain variations
common_providers = {
    # Google
    "gmail.com": "Google",
    "googlemail.com": "Google",
    "google.com": "Google",
    # Microsoft
    "outlook.com": "Microsoft",
    "hotmail.com": "Microsoft",
    "live.com": "Microsoft",
    "msn.com": "Microsoft",
    "office365.com": "Microsoft",
    "outlook.co": "Microsoft",
    "hotmail.co": "Microsoft",
    # Yahoo
    "yahoo.com": "Yahoo",
    "ymail.com": "Yahoo",
    "rocketmail.com": "Yahoo",
    "yahoo.co": "Yahoo",
    # AOL
    "aol.com": "AOL",
    "aim.com": "AOL",
    # Apple
    "icloud.com": "Apple",
    "me.com": "Apple",
    "mac.com": "Apple",
    # ProtonMail
    "protonmail.com": "ProtonMail",
    "proton.me": "ProtonMail",
    # Zoho
    "zoho.com": "Zoho",
    "zohomail.com": "Zoho",
    # Others
    "mail.com": "Mail.com",
    "gmx.com": "GMX",
    "gmx.net": "GMX",
    "fastmail.com": "FastMail",
    "tutanota.com": "Tutanota",
}

# MX-based provider detection for common providers
mx_provider_map = {
    "googlemail.l.google.com": "Google",
    "google.com": "Google",
    "aspmx.l.google.com": "Google",
    "mx.microsoft.com": "Microsoft",
    "outlook.com": "Microsoft",
    "mx1.hotmail.com": "Microsoft",
    "mx2.hotmail.com": "Microsoft",
    "mx3.hotmail.com": "Microsoft",
    "mx4.hotmail.com": "Microsoft",
    "mx1.mail.yahoo.com": "Yahoo",
    "mx2.mail.yahoo.com": "Yahoo",
    "mx5.mail.yahoo.com": "Yahoo",
    "mx-aol.mail.gm0.yahoodns.net": "AOL",
    "mx1.mail.icloud.com": "Apple",
    "mx2.mail.icloud.com": "Apple",
    "mx3.mail.icloud.com": "Apple",
    "mx.protonmail.ch": "ProtonMail",
    "mx1.zoho.com": "Zoho",
    "mx2.zoho.com": "Zoho",
    "mx3.zoho.com": "Zoho",
    "mx.gmx.com": "GMX",
    "mx.fastmail.com": "FastMail",
    "mx.tutanota.de": "Tutanota",
}


# ------------- UTIL FUNCTIONS -----------------

def is_valid_format(email):
    try:
        validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


async def has_mx_record(domain):
    try:
        resolver = aiodns.DNSResolver(timeout=2.0)
        answers = await resolver.query(domain, 'MX')
        mx_records = [str(answer.host).lower() for answer in answers]
        return True, mx_records
    except Exception as e:
        logger.warning(f"MX query failed for {domain}: {str(e)}")
        return False, []


def detect_provider(domain, mx_records=None):
    domain = domain.lower().strip()
    if not domain:
        return "Other"

    # Direct domain match
    if domain in common_providers:
        return common_providers[domain]

    # Check for domain variations (e.g., google.co.uk)
    for provider_domain, provider in common_providers.items():
        if provider_domain in domain or domain.endswith(f".{provider_domain}"):
            return provider

    # Fallback to MX record-based detection
    if mx_records:
        for mx in mx_records:
            for mx_key, provider in mx_provider_map.items():
                if mx_key in mx or mx.endswith(f".{mx_key}"):
                    return provider

    return "Other"


def detect_country(domain):
    tld = domain.split('.')[-1] if '.' in domain else ''
    return tld.upper() if len(tld) == 2 else 'INTL'


def contains_banned_keywords(email):
    email_lower = email.lower()
    return any(bad_word in email_lower for bad_word in banned_keywords)


async def validate_batch(emails, domains, mx_cache):
    results = []
    # Pre-validate email formats
    format_results = [(email, is_valid_format(email)) for email in emails]
    valid_emails = [(email, valid) for email, valid in format_results if valid]

    # Get unique domains from valid emails
    valid_domains = set(email.partition("@")[2] for email, _ in valid_emails)

    # Batch DNS queries with error handling
    async def query_mx(domain):
        try:
            if domain in mx_cache:
                return domain, *mx_cache[domain]
            result, mx_records = await has_mx_record(domain)
            return domain, result, mx_records
        except Exception as e:
            logger.error(f"Error querying MX for {domain}: {str(e)}")
            return domain, False, []

    mx_tasks = [query_mx(domain) for domain in valid_domains]
    mx_results = await asyncio.gather(*mx_tasks, return_exceptions=True)

    # Update MX cache
    for result in mx_results:
        if not isinstance(result, Exception):
            domain, mx_valid, mx_records = result
            mx_cache[domain] = (mx_valid, mx_records)
        else:
            logger.error(f"Exception in MX query: {str(result)}")

    # Process results
    for email, valid_format in format_results:
        if not valid_format:
            results.append((email, False, False))
            continue
        domain = email.partition("@")[2]
        mx_valid, mx_records = mx_cache.get(domain, (False, []))
        results.append((email, valid_format, mx_valid))

    return results


async def process_emails(df, batch_size=100):
    emails = df['Email'].tolist()
    domains = [email.partition("@")[2] for email in emails]
    output = []
    total = len(emails)
    progress = st.progress(0, text="Validating Emails... 0%")
    mx_cache = {}

    for i in range(0, total, batch_size):
        start_time = time.time()
        batch = emails[i:i + batch_size]
        batch_domains = domains[i:i + batch_size]
        results = await validate_batch(batch, batch_domains, mx_cache)

        for email, valid_format, mx_valid in results:
            local, _, domain = email.partition("@")
            mx_records = mx_cache.get(domain, (False, []))[1]
            provider = detect_provider(domain, mx_records)
            country = detect_country(domain)
            bad = contains_banned_keywords(email)
            output.append({
                "Email": email,
                "ValidFormat": valid_format,
                "MXValid": mx_valid,
                "Provider": provider,
                "Country": country,
                "BadEmail": bad
            })

        done = min(i + batch_size, total)
        percent = int((done / total) * 100)
        progress.progress(done / total, text=f"✅ Processed {done}/{total} emails ({percent}%)")
        logger.info(f"Batch {i // batch_size + 1} took {time.time() - start_time:.2f} seconds")

    return output


# ------------- MAIN APP -----------------

uploaded = st.file_uploader("📤 Upload your CSV (must contain `email` column)", type=["csv"])

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]

        if "email" not in df_raw.columns:
            st.error("❌ CSV must contain an 'email' column.")
        else:
            df_raw.rename(columns={"email": "Email"}, inplace=True)

            original_count = len(df_raw)
            df_raw['Email'] = df_raw['Email'].astype(str).str.strip()
            duplicate_count = df_raw.duplicated(subset='Email').sum()
            df_raw.drop_duplicates(subset='Email', inplace=True)

            st.info(f"📊 Found {original_count} emails — 🧹 Removed {duplicate_count} duplicates.")

            results = asyncio.run(process_emails(df_raw))
            df = pd.DataFrame(results)

            # Filter only valid & non-bad emails
            df = df[~df['BadEmail'] & df['ValidFormat'] & df['MXValid']]
            df.drop_duplicates(subset='Email', inplace=True)

            st.session_state['processed_df'] = df
    except Exception as e:
        st.error(f"❌ Error processing CSV: {str(e)}")
        logger.error(f"CSV processing error: {str(e)}")

# ------------- DISPLAY + DOWNLOADS -----------------

if 'processed_df' in st.session_state:
    df = st.session_state['processed_df']

    st.subheader("📊 Email Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Total Valid Emails", len(df))
    col2.metric("📦 Unique Providers", df['Provider'].nunique())
    col3.metric("🌍 Countries", df['Country'].nunique())

    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("#### 📡 Provider Distribution")
        provider_counts = df['Provider'].value_counts().to_dict()
        st.bar_chart(provider_counts)
    with chart2:
        st.markdown("#### 🌎 Country Distribution")
        country_counts = df['Country'].value_counts().to_dict()
        st.bar_chart(country_counts)

    st.subheader("🔍 Search & Explore Table")
    search = st.text_input("🔎 Search emails or domains")
    if search:
        df_filtered = df[df['Email'].str.contains(search, case=False, na=False) |
                         df['Email'].str.contains(search.split("@")[-1], case=False, na=False)]
    else:
        df_filtered = df

    st.dataframe(df_filtered, use_container_width=True)

    st.subheader("📁 Download by Filter")

    colA, colB = st.columns(2)
    with colA:
        selected_provider = st.selectbox("🎯 Select Provider", sorted(df['Provider'].unique()))
        df_provider = df[df['Provider'] == selected_provider]
        st.download_button(
            label=f"⬇️ Download {selected_provider} Emails",
            data=df_provider.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_provider}_emails.csv",
            mime='text/csv'
        )

    with colB:
        selected_country = st.selectbox("🌐 Select Country", sorted(df['Country'].unique()))
        df_country = df[df['Country'] == selected_country]
        st.download_button(
            label=f"⬇️ Download {selected_country} Emails",
            data=df_country.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_country}_emails.csv",
            mime='text/csv'
        )