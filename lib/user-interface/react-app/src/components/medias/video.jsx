
import React from 'react';
import { useEffect, useState } from "react";
import "../../styles/new-components.css"

function getYoutubeThumbnailFromId(id) {
    return id ? `https://img.youtube.com/vi/${id}/hqdefault.jpg` : null
}

export default function oneVideo({ url, id }) {

    const thumbnail = getYoutubeThumbnailFromId(id)
    const [title, setTitle] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            const data = await fetch(
                `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${id}&format=json`,
            );
            const json = await data.json();
            setTitle(json.title)
        };
        fetchData()
    }, [url]);

    console.log(title)

    return (
        <a href={url} target="_blank"
            className="link_element"
            style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                padding: "0.5rem",
                gap: "0.5rem",
                width: "12rem"
            }}
        >
            <img
                src={thumbnail}
                style={{
                    width: "8rem",
                    objectFit: "contain",
                    marginLeft: "auto",
                    marginRight: "auto"
                }}
            />
            <div style={{
                position: "relative",
                height: "10rem",
                display: "inline-block",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "wrap",
                textAlign: "center",
                fontWeight: "600"
            }}>{title}</div>
        </a>
    )
}

