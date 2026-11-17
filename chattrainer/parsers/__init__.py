from . import discord, instagram, snapchat, telegram, tiktok, whatsapp

PARSERS = {
    "whatsapp": whatsapp.parse,
    "instagram": instagram.parse,
    "tiktok": tiktok.parse,
    "telegram": telegram.parse,
    "discord": discord.parse,
    "snapchat": snapchat.parse,
}
