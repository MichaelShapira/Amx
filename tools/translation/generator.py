"""Seeded synthetic parallel corpus for the translation use case: English, German and Spanish.

Every item is written three times from the same template, one per language, so the three
texts say exactly the same thing and any of them can serve as the reference translation of
another. Templates are hand-written sentence frames per domain (customer support, logistics,
HR notices, travel confirmations, product release notes, invoices); slots hold names, cities,
dates, times, amounts and quantities, each rendered the way that language writes it (14. März
2026 / March 14, 2026 / 14 de marzo de 2026; 1.250,00 € / €1,250.00; Munich / München / Múnich).
Nothing is real: names, order numbers and amounts are made up, and nothing depends on the
current date or on Python's hash randomization, so the same seed gives the same corpus.

Two sizes: a "sentence" item is one or two body sentences; a "paragraph" item is an opening
line, three to five body sentences and a closing line of one domain, with consistent slot
values throughout (the same person, order and city).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

LANGUAGES = ("en", "de", "es")
LANGUAGE_NAMES = {"en": {"en": "English", "de": "Englisch", "es": "inglés"},
                  "de": {"en": "German", "de": "Deutsch", "es": "alemán"},
                  "es": {"en": "Spanish", "de": "Spanisch", "es": "español"}}
DIRECTIONS = ("en-de", "de-en", "en-es", "es-en")
SIZES = ("sentence", "paragraph")


def derive_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Frame:
    role: str          # open | body | close
    en: str
    de: str
    es: str

    def text(self, lang: str) -> str:
        return getattr(self, lang)


def _d(role: str, en: str, de: str, es: str) -> Frame:
    return Frame(role, en, de, es)


# ---------------------------------------------------------------------------- frames
# {days_dat} is the German dative plural ("innerhalb von 3 Tagen"); English and Spanish use {days}.
DOMAINS: dict[str, list[Frame]] = {
    "support": [
        _d("open", "Hello {name}, thank you for contacting our support team.",
           "Guten Tag {name}, vielen Dank, dass Sie sich an unser Support-Team gewandt haben.",
           "Hola, {name}: gracias por ponerse en contacto con nuestro equipo de soporte."),
        _d("open", "Hello {first}, we have received your message about order {order}.",
           "Hallo {first}, wir haben Ihre Nachricht zur Bestellung {order} erhalten.",
           "Hola, {first}: hemos recibido su mensaje sobre el pedido {order}."),
        _d("body", "Your order {order} was shipped on {date} and should arrive within {days}.",
           "Ihre Bestellung {order} wurde am {date} versandt und sollte innerhalb von {days_dat} eintreffen.",
           "Su pedido {order} se envió el {date} y debería llegar en un plazo de {days}."),
        _d("body", "We have refunded {amount} to your original payment method.",
           "Wir haben {amount} auf Ihre ursprüngliche Zahlungsmethode zurückerstattet.",
           "Hemos reembolsado {amount} a su método de pago original."),
        _d("body", "The replacement part will be sent from our warehouse in {city} free of charge.",
           "Das Ersatzteil wird kostenlos aus unserem Lager in {city} verschickt.",
           "La pieza de repuesto se enviará sin coste desde nuestro almacén en {city}."),
        _d("body", "Please keep the original packaging until the inspection is complete.",
           "Bitte bewahren Sie die Originalverpackung auf, bis die Prüfung abgeschlossen ist.",
           "Por favor, conserve el embalaje original hasta que se complete la inspección."),
        _d("body", "Our records show that the device was registered on {date}.",
           "Unseren Unterlagen zufolge wurde das Gerät am {date} registriert.",
           "Según nuestros registros, el dispositivo se registró el {date}."),
        _d("body", "The warranty covers repairs for {months} from the date of purchase.",
           "Die Garantie deckt Reparaturen für {months} ab dem Kaufdatum ab.",
           "La garantía cubre las reparaciones durante {months} a partir de la fecha de compra."),
        _d("body", "A technician can visit you on {weekday} between {time} and {time2}.",
           "Ein Techniker kann Sie am {weekday} zwischen {time} und {time2} Uhr besuchen.",
           "Un técnico puede visitarle el {weekday} entre las {time} y las {time2}."),
        _d("body", "If the problem persists, reply to this message with a short video of the error.",
           "Sollte das Problem weiterhin bestehen, antworten Sie auf diese Nachricht mit einem kurzen Video des Fehlers.",
           "Si el problema persiste, responda a este mensaje con un vídeo breve del error."),
        _d("body", "We apologize for the inconvenience this has caused.",
           "Wir entschuldigen uns für die entstandenen Unannehmlichkeiten.",
           "Le pedimos disculpas por las molestias ocasionadas."),
        _d("close", "Kind regards, {agent} from the customer care team.",
           "Mit freundlichen Grüßen, {agent} vom Kundenservice-Team.",
           "Un cordial saludo, {agent}, del equipo de atención al cliente."),
        _d("close", "You can reach us every day from {time} to {time2}.",
           "Sie erreichen uns täglich von {time} bis {time2} Uhr.",
           "Puede contactarnos todos los días de {time} a {time2}."),
    ],
    "logistics": [
        _d("open", "Status update for shipment {order}.",
           "Statusaktualisierung für Sendung {order}.",
           "Actualización de estado del envío {order}."),
        _d("open", "This message concerns the delivery scheduled for {date}.",
           "Diese Nachricht betrifft die für den {date} geplante Lieferung.",
           "Este mensaje se refiere a la entrega programada para el {date}."),
        _d("body", "The truck left the distribution center in {city} at {time}.",
           "Der Lkw hat das Verteilzentrum in {city} um {time} Uhr verlassen.",
           "El camión salió del centro de distribución de {city} a las {time}."),
        _d("body", "{count} pallets were loaded; two of them require cold storage.",
           "{count} Paletten wurden geladen; zwei davon benötigen Kühllagerung.",
           "Se cargaron {count} palés; dos de ellos requieren almacenamiento en frío."),
        _d("body", "Customs clearance in {city2} took {hours} longer than planned.",
           "Die Zollabfertigung in {city2} dauerte {hours} länger als geplant.",
           "El despacho de aduanas en {city2} tardó {hours} más de lo previsto."),
        _d("body", "The estimated time of arrival is now {weekday} at {time2}.",
           "Die voraussichtliche Ankunftszeit ist nun {weekday} um {time2} Uhr.",
           "La hora estimada de llegada es ahora el {weekday} a las {time2}."),
        _d("body", "Please make sure a forklift is available at the loading dock.",
           "Bitte stellen Sie sicher, dass an der Laderampe ein Gabelstapler bereitsteht.",
           "Por favor, asegúrese de que haya una carretilla elevadora disponible en el muelle de carga."),
        _d("body", "The total weight of the shipment is {weight}.",
           "Das Gesamtgewicht der Sendung beträgt {weight}.",
           "El peso total del envío es de {weight}."),
        _d("body", "Any damage must be noted on the delivery receipt before signing.",
           "Schäden müssen vor der Unterschrift auf dem Lieferschein vermerkt werden.",
           "Cualquier daño debe anotarse en el albarán de entrega antes de firmar."),
        _d("body", "The driver will call the contact person {minutes} before arrival.",
           "Der Fahrer ruft die Kontaktperson {minutes} vor der Ankunft an.",
           "El conductor llamará a la persona de contacto {minutes} antes de la llegada."),
        _d("body", "Tracking details are updated every {hours} on the customer portal.",
           "Die Sendungsverfolgung wird alle {hours} im Kundenportal aktualisiert.",
           "Los detalles de seguimiento se actualizan cada {hours} en el portal del cliente."),
        _d("close", "Thank you for your patience, the logistics team.",
           "Vielen Dank für Ihre Geduld, das Logistikteam.",
           "Gracias por su paciencia, el equipo de logística."),
        _d("close", "For questions about this shipment, quote reference {order}.",
           "Bei Fragen zu dieser Sendung geben Sie bitte die Referenz {order} an.",
           "Para consultas sobre este envío, indique la referencia {order}."),
    ],
    "hr": [
        _d("open", "Dear colleagues, here is a short update from the human resources department.",
           "Liebe Kolleginnen und Kollegen, hier ist ein kurzes Update aus der Personalabteilung.",
           "Estimados compañeros y compañeras, aquí tienen una breve actualización del departamento de recursos humanos."),
        _d("open", "Hello everyone, please read the following information carefully.",
           "Hallo zusammen, bitte lesen Sie die folgenden Informationen sorgfältig durch.",
           "Hola a todos, lean atentamente la siguiente información."),
        _d("body", "The office in {city} will be closed on {weekday}, {date}, for maintenance work.",
           "Das Büro in {city} bleibt am {weekday}, dem {date}, wegen Wartungsarbeiten geschlossen.",
           "La oficina de {city} permanecerá cerrada el {weekday} {date} por trabajos de mantenimiento."),
        _d("body", "{name} joins the team on {date} to lead the project.",
           "{name} kommt am {date} ins Team, um das Projekt zu leiten.",
           "{name} se incorpora al equipo el {date} para dirigir el proyecto."),
        _d("body", "Vacation requests for the summer must be submitted by {date}.",
           "Urlaubsanträge für den Sommer müssen bis zum {date} eingereicht werden.",
           "Las solicitudes de vacaciones para el verano deben presentarse antes del {date}."),
        _d("body", "The training on data protection takes {hours} and is mandatory for all employees.",
           "Die Schulung zum Datenschutz dauert {hours} und ist für alle Mitarbeitenden verpflichtend.",
           "La formación sobre protección de datos dura {hours} y es obligatoria para todos los empleados."),
        _d("body", "Starting next month, the cafeteria opens at {time} instead of {time2}.",
           "Ab nächstem Monat öffnet die Kantine um {time} Uhr statt um {time2} Uhr.",
           "A partir del próximo mes, la cafetería abre a las {time} en lugar de a las {time2}."),
        _d("body", "Travel expenses are reimbursed within {days} of submitting the receipts.",
           "Reisekosten werden innerhalb von {days_dat} nach Einreichung der Belege erstattet.",
           "Los gastos de viaje se reembolsan en un plazo de {days} desde la presentación de los recibos."),
        _d("body", "The annual survey closes on {weekday}; so far {percent} of the staff have responded.",
           "Die jährliche Umfrage endet am {weekday}; bisher haben {percent} der Belegschaft geantwortet.",
           "La encuesta anual se cierra el {weekday}; hasta ahora ha respondido el {percent} del personal."),
        _d("body", "Parking spaces in the underground garage can be booked through the intranet.",
           "Parkplätze in der Tiefgarage können über das Intranet gebucht werden.",
           "Las plazas de aparcamiento del garaje subterráneo se pueden reservar a través de la intranet."),
        _d("body", "Please update your emergency contact details in the employee portal.",
           "Bitte aktualisieren Sie Ihre Notfallkontaktdaten im Mitarbeiterportal.",
           "Por favor, actualicen sus datos de contacto de emergencia en el portal del empleado."),
        _d("close", "Best regards, {agent} on behalf of the HR team.",
           "Beste Grüße, {agent} im Namen des HR-Teams.",
           "Saludos cordiales, {agent}, en nombre del equipo de RR. HH."),
        _d("close", "Questions can be sent to the HR mailbox at any time.",
           "Fragen können jederzeit an das HR-Postfach gesendet werden.",
           "Las preguntas pueden enviarse en cualquier momento al buzón de RR. HH."),
    ],
    "travel": [
        _d("open", "Your booking {order} is confirmed.",
           "Ihre Buchung {order} ist bestätigt.",
           "Su reserva {order} está confirmada."),
        _d("open", "Thank you for booking with us, {first}.",
           "Vielen Dank für Ihre Buchung bei uns, {first}.",
           "Gracias por reservar con nosotros, {first}."),
        _d("body", "The flight departs from {city} on {date} at {time} and lands in {city2} at {time2}.",
           "Der Flug startet am {date} um {time} Uhr in {city} und landet um {time2} Uhr in {city2}.",
           "El vuelo sale de {city} el {date} a las {time} y aterriza en {city2} a las {time2}."),
        _d("body", "Check-in opens {hours} before departure and closes {minutes} before departure.",
           "Der Check-in öffnet {hours} vor dem Abflug und schließt {minutes} vor dem Abflug.",
           "La facturación abre {hours} antes de la salida y cierra {minutes} antes de la salida."),
        _d("body", "Your hotel room is reserved for {nights} with breakfast included.",
           "Ihr Hotelzimmer ist für {nights} mit Frühstück reserviert.",
           "Su habitación de hotel está reservada para {nights} con desayuno incluido."),
        _d("body", "A shuttle to the hotel is available every {minutes} from terminal 2.",
           "Ein Shuttle zum Hotel fährt alle {minutes} ab Terminal 2.",
           "Hay un servicio de traslado al hotel cada {minutes} desde la terminal 2."),
        _d("body", "The total price of {amount} has been charged to your card.",
           "Der Gesamtpreis von {amount} wurde Ihrer Karte belastet.",
           "El precio total de {amount} se ha cargado a su tarjeta."),
        _d("body", "Cancellations are free of charge up to {days} before arrival.",
           "Stornierungen sind bis {days} vor der Anreise kostenlos.",
           "Las cancelaciones son gratuitas hasta {days} antes de la llegada."),
        _d("body", "Each passenger may bring one carry-on bag of up to {weight_small}.",
           "Jeder Fluggast darf ein Handgepäckstück mit bis zu {weight_small} mitnehmen.",
           "Cada pasajero puede llevar una pieza de equipaje de mano de hasta {weight_small}."),
        _d("body", "The rental car can be picked up at the airport counter from {time}.",
           "Der Mietwagen kann ab {time} Uhr am Flughafenschalter abgeholt werden.",
           "El coche de alquiler se puede recoger en el mostrador del aeropuerto a partir de las {time}."),
        _d("body", "Please bring a valid passport and the booking reference to the airport.",
           "Bitte bringen Sie einen gültigen Reisepass und die Buchungsreferenz zum Flughafen mit.",
           "Por favor, lleve al aeropuerto un pasaporte válido y la referencia de la reserva."),
        _d("close", "We wish you a pleasant trip.",
           "Wir wünschen Ihnen eine angenehme Reise.",
           "Le deseamos un buen viaje."),
        _d("close", "Your travel documents are attached to this message.",
           "Ihre Reiseunterlagen sind dieser Nachricht beigefügt.",
           "Sus documentos de viaje se adjuntan a este mensaje."),
    ],
    "product": [
        _d("open", "Release notes for {product}, version {version}.",
           "Versionshinweise für {product}, Version {version}.",
           "Notas de la versión de {product}, versión {version}."),
        _d("open", "This update to {product} is available from {date}.",
           "Dieses Update für {product} ist ab dem {date} verfügbar.",
           "Esta actualización de {product} está disponible a partir del {date}."),
        _d("body", "The new battery lasts up to {hours} on a single charge.",
           "Der neue Akku hält mit einer einzigen Ladung bis zu {hours}.",
           "La nueva batería dura hasta {hours} con una sola carga."),
        _d("body", "Start-up time has been reduced by {percent} compared with the previous version.",
           "Die Startzeit wurde im Vergleich zur vorherigen Version um {percent} verkürzt.",
           "El tiempo de arranque se ha reducido un {percent} en comparación con la versión anterior."),
        _d("body", "The device weighs {weight_small} and fits in a standard backpack.",
           "Das Gerät wiegt {weight_small} und passt in einen gewöhnlichen Rucksack.",
           "El dispositivo pesa {weight_small} y cabe en una mochila estándar."),
        _d("body", "{product} now supports {count} languages in the user interface.",
           "{product} unterstützt jetzt {count} Sprachen in der Benutzeroberfläche.",
           "{product} ahora admite {count} idiomas en la interfaz de usuario."),
        _d("body", "A known issue with the Bluetooth connection will be fixed in the next release.",
           "Ein bekanntes Problem mit der Bluetooth-Verbindung wird in der nächsten Version behoben.",
           "Un problema conocido con la conexión Bluetooth se corregirá en la próxima versión."),
        _d("body", "The software update is downloaded automatically when the device is connected to Wi-Fi.",
           "Das Software-Update wird automatisch heruntergeladen, wenn das Gerät mit dem WLAN verbunden ist.",
           "La actualización de software se descarga automáticamente cuando el dispositivo está conectado a la red wifi."),
        _d("body", "Customers in {city} can test the device in our showroom from {weekday}.",
           "Kunden in {city} können das Gerät ab {weekday} in unserem Showroom testen.",
           "Los clientes de {city} pueden probar el dispositivo en nuestra sala de exposición a partir del {weekday}."),
        _d("body", "The recommended retail price is {amount}, including taxes.",
           "Die unverbindliche Preisempfehlung beträgt {amount} inklusive Steuern.",
           "El precio de venta recomendado es de {amount}, impuestos incluidos."),
        _d("body", "All settings are backed up to the cloud every {minutes}.",
           "Alle Einstellungen werden alle {minutes} in der Cloud gesichert.",
           "Todos los ajustes se guardan en la nube cada {minutes}."),
        _d("close", "The full changelog is available in the help center.",
           "Das vollständige Änderungsprotokoll finden Sie im Hilfecenter.",
           "El registro completo de cambios está disponible en el centro de ayuda."),
        _d("close", "Thank you for choosing {product}.",
           "Vielen Dank, dass Sie sich für {product} entschieden haben.",
           "Gracias por elegir {product}."),
    ],
    "finance": [
        _d("open", "Invoice {order} is now available in your account.",
           "Die Rechnung {order} ist jetzt in Ihrem Konto verfügbar.",
           "La factura {order} ya está disponible en su cuenta."),
        _d("open", "This is a reminder about your payment due on {date}.",
           "Dies ist eine Erinnerung an Ihre am {date} fällige Zahlung.",
           "Este es un recordatorio sobre su pago con vencimiento el {date}."),
        _d("body", "The outstanding amount of {amount} is due within {days}.",
           "Der offene Betrag von {amount} ist innerhalb von {days_dat} fällig.",
           "El importe pendiente de {amount} debe abonarse en un plazo de {days}."),
        _d("body", "A discount of {percent} applies if the invoice is paid before {date}.",
           "Bei Zahlung vor dem {date} gilt ein Rabatt von {percent}.",
           "Se aplica un descuento del {percent} si la factura se paga antes del {date}."),
        _d("body", "Payments by bank transfer are usually credited within {days}.",
           "Zahlungen per Überweisung werden in der Regel innerhalb von {days_dat} gutgeschrieben.",
           "Los pagos por transferencia bancaria suelen abonarse en un plazo de {days}."),
        _d("body", "Please quote the invoice number in the payment reference.",
           "Bitte geben Sie die Rechnungsnummer im Verwendungszweck an.",
           "Por favor, indique el número de factura en el concepto del pago."),
        _d("body", "The subscription renews automatically on {date} at a price of {amount} per year.",
           "Das Abonnement verlängert sich am {date} automatisch zum Preis von {amount} pro Jahr.",
           "La suscripción se renueva automáticamente el {date} a un precio de {amount} al año."),
        _d("body", "Late payments incur a fee of {amount_small} per reminder.",
           "Für verspätete Zahlungen wird eine Gebühr von {amount_small} pro Mahnung erhoben.",
           "Los pagos atrasados conllevan un cargo de {amount_small} por recordatorio."),
        _d("body", "The monthly statement lists {count} transactions in total.",
           "Die Monatsabrechnung weist insgesamt {count} Transaktionen aus.",
           "El extracto mensual recoge un total de {count} transacciones."),
        _d("body", "Our finance office in {city} is open on weekdays from {time} to {time2}.",
           "Unsere Finanzabteilung in {city} ist werktags von {time} bis {time2} Uhr geöffnet.",
           "Nuestra oficina de finanzas en {city} abre los días laborables de {time} a {time2}."),
        _d("body", "The exchange rate used for this invoice was fixed on {date}.",
           "Der für diese Rechnung verwendete Wechselkurs wurde am {date} festgelegt.",
           "El tipo de cambio utilizado para esta factura se fijó el {date}."),
        _d("close", "Thank you for your prompt payment.",
           "Vielen Dank für Ihre pünktliche Zahlung.",
           "Gracias por su pronto pago."),
        _d("close", "If you have already paid, please disregard this notice.",
           "Falls Sie bereits bezahlt haben, betrachten Sie dieses Schreiben bitte als gegenstandslos.",
           "Si ya ha realizado el pago, ignore este aviso."),
    ],
}

# ---------------------------------------------------------------------------- slot values
FIRST_NAMES = ["Anna", "Lukas", "Sofia", "Daniel", "Elena", "Jonas", "Marta", "Tobias", "Clara", "Pablo",
               "Laura", "Felix", "Carmen", "Leon", "Julia", "Mateo", "Hanna", "Diego", "Nina", "Oscar"]
LAST_NAMES = ["Keller", "García", "Brandt", "Moreno", "Fischer", "Ruiz", "Weber", "Navarro", "Hoffmann", "Serrano",
              "Schulz", "Castro", "Lange", "Ortega", "Vogel", "Molina", "Krüger", "Delgado", "Winter", "Romero"]
# English / German / Spanish forms of the same place: a translator must localize these.
CITIES = [("Munich", "München", "Múnich"), ("Vienna", "Wien", "Viena"), ("Cologne", "Köln", "Colonia"),
          ("Lisbon", "Lissabon", "Lisboa"), ("Seville", "Sevilla", "Sevilla"), ("Zurich", "Zürich", "Zúrich"),
          ("Milan", "Mailand", "Milán"), ("Prague", "Prag", "Praga"), ("Brussels", "Brüssel", "Bruselas"),
          ("Warsaw", "Warschau", "Varsovia"), ("Hamburg", "Hamburg", "Hamburgo"), ("Madrid", "Madrid", "Madrid"),
          ("Barcelona", "Barcelona", "Barcelona"), ("Berlin", "Berlin", "Berlín"), ("Geneva", "Genf", "Ginebra"),
          ("Copenhagen", "Kopenhagen", "Copenhague"), ("Rotterdam", "Rotterdam", "Róterdam"),
          ("Frankfurt", "Frankfurt", "Fráncfort"), ("Valencia", "Valencia", "Valencia"), ("Porto", "Porto", "Oporto")]
WEEKDAYS = [("Monday", "Montag", "lunes"), ("Tuesday", "Dienstag", "martes"), ("Wednesday", "Mittwoch", "miércoles"),
            ("Thursday", "Donnerstag", "jueves"), ("Friday", "Freitag", "viernes"), ("Saturday", "Samstag", "sábado")]
MONTHS = [("January", "Januar", "enero"), ("February", "Februar", "febrero"), ("March", "März", "marzo"),
          ("April", "April", "abril"), ("May", "Mai", "mayo"), ("June", "Juni", "junio"), ("July", "Juli", "julio"),
          ("August", "August", "agosto"), ("September", "September", "septiembre"), ("October", "Oktober", "octubre"),
          ("November", "November", "noviembre"), ("December", "Dezember", "diciembre")]
PRODUCTS = ["Aurora X2", "Nimbus Pro", "Vega 400", "Helix Mini", "Orion Tab", "Lumen Hub"]
LANG_INDEX = {"en": 0, "de": 1, "es": 2}


def _group(n: int, lang: str) -> str:
    """Thousands separator: 1,250 in English, 1.250 in German and Spanish; none below 1,000."""
    s = f"{n:,}"
    return s if lang == "en" else s.replace(",", ".")


def _money(euros: int, cents: int, lang: str) -> str:
    if lang == "en":
        return f"€{euros:,}.{cents:02d}"
    return f"{_group(euros, lang)},{cents:02d} €"


def _unit(k: int, lang: str, en: str, de: str, es: str) -> str:
    return f"{k} {dict(en=en, de=de, es=es)[lang]}"


def _time(h: int, m: int, lang: str) -> str:
    """German writes 09:30 (the frames add "Uhr"); English and Spanish write 9:30."""
    return f"{h:02d}:{m:02d}" if lang == "de" else f"{h}:{m:02d}"


def _date(day: int, month: int, year: int, lang: str) -> str:
    mon = MONTHS[month - 1][LANG_INDEX[lang]]
    if lang == "en":
        return f"{mon} {day}, {year}"
    if lang == "de":
        return f"{day}. {mon} {year}"
    return f"{day} de {mon} de {year}"


@dataclass
class Slots:
    """One item's slot values, rendered per language on demand."""
    raw: dict

    def render(self, lang: str) -> dict[str, str]:
        r = self.raw
        i = LANG_INDEX[lang]
        return {
            "name": f"{r['first']} {r['last']}", "first": r["first"], "agent": r["agent"],
            "city": CITIES[r["city"]][i], "city2": CITIES[r["city2"]][i],
            "weekday": WEEKDAYS[r["weekday"]][i],
            "date": _date(r["day"], r["month"], r["year"], lang),
            "time": _time(*r["time"], lang), "time2": _time(*r["time2"], lang),
            "amount": _money(r["euros"], r["cents"], lang), "amount_small": _money(r["euros_small"], 0, lang),
            "percent": f"{r['percent']}%" if lang == "en" else f"{r['percent']} %",
            "count": str(r["count"]),
            "days": _unit(r["days"], lang, "days", "Tage", "días"), "days_dat": f"{r['days']} Tagen",
            "hours": _unit(r["hours"], lang, "hours", "Stunden", "horas"),
            "minutes": _unit(r["minutes"], lang, "minutes", "Minuten", "minutos"),
            "months": _unit(r["months"], lang, "months", "Monate", "meses"),
            "nights": _unit(r["nights"], lang, "nights", "Nächte", "noches"),
            "weight": f"{_group(r['weight'], lang)} kg", "weight_small": f"{r['weight_small']} kg",
            "order": r["order"], "product": r["product"], "version": r["version"],
        }


def make_slots(rng: random.Random) -> Slots:
    first, agent = rng.sample(FIRST_NAMES, 2)
    city, city2 = rng.sample(range(len(CITIES)), 2)
    h1 = rng.randrange(7, 15)
    h2 = rng.randrange(h1 + 1, 21)
    return Slots({
        "first": first, "last": rng.choice(LAST_NAMES), "agent": agent, "city": city, "city2": city2,
        "weekday": rng.randrange(len(WEEKDAYS)), "day": rng.randrange(1, 29), "month": rng.randrange(1, 13),
        "year": rng.choice([2026, 2027]), "time": (h1, rng.choice([0, 15, 30, 45])), "time2": (h2, rng.choice([0, 30])),
        "euros": rng.choice([49, 120, 349, 1250, 2480, 15600]), "cents": rng.choice([0, 50, 90, 99]),
        "euros_small": rng.choice([5, 10, 15, 25]), "percent": rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 68, 85]),
        "count": rng.randrange(3, 49), "days": rng.randrange(2, 15), "hours": rng.randrange(2, 13),
        "minutes": rng.choice([10, 15, 20, 30, 45]), "months": rng.choice([12, 24, 36]), "nights": rng.randrange(2, 8),
        "weight": rng.choice([250, 480, 760, 1250, 1800, 2400]), "weight_small": rng.randrange(1, 10),
        "order": f"{rng.choice(['ORD', 'INV', 'SHP', 'BK'])}-{rng.randrange(2026, 2028)}-{rng.randrange(1000, 9999)}",
        "product": rng.choice(PRODUCTS), "version": f"{rng.randrange(1, 7)}.{rng.randrange(0, 10)}.{rng.randrange(0, 10)}",
    })


# ---------------------------------------------------------------------------- items
def _pick(rng: random.Random, frames: list[Frame], size: str) -> list[Frame]:
    body = [f for f in frames if f.role == "body"]
    if size == "sentence":
        return rng.sample(body, rng.choice([1, 2]))
    opening = rng.choice([f for f in frames if f.role == "open"])
    closing = rng.choice([f for f in frames if f.role == "close"])
    return [opening, *rng.sample(body, rng.randrange(3, 6)), closing]


def render_item(frames: list[Frame], slots: Slots) -> dict[str, str]:
    """The same frames filled in each language; sentences joined with single spaces."""
    return {lang: " ".join(f.text(lang).format(**slots.render(lang)) for f in frames) for lang in LANGUAGES}


def build_items(n_per_size: int, seed: int) -> list[dict]:
    """n_per_size items of each size, cycling through the domains so every domain is equally represented."""
    out: list[dict] = []
    domains = sorted(DOMAINS)
    for size in SIZES:
        for k in range(n_per_size):
            domain = domains[k % len(domains)]
            rng = random.Random(derive_seed(seed, size, k))  # noqa: S311 - reproducible synthetic text, not security
            frames = _pick(rng, DOMAINS[domain], size)
            text = render_item(frames, make_slots(rng))
            out.append({"id": f"t-{size[0]}-{k:04d}", "size": size, "domain": domain, "text": text,
                        "chars": {lang: len(t) for lang, t in text.items()},
                        "words": {lang: len(t.split()) for lang, t in text.items()},
                        "sentences": len(frames)})
    return out


def frame_count() -> int:
    return sum(len(v) for v in DOMAINS.values())
